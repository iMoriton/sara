#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Hugging Face Transformers Trainer + DeepSpeed 进行指令微调训练
支持模型自带的 chat template
"""
from functools import partial
import os
import json
import argparse
from typing import Dict, List, Optional, Any
import wandb
import torch
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.integrations import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import torch.distributed as dist
from pathlib import Path
import numpy as np
import deepspeed

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0" # H100

class CustomDataCollator(DataCollatorForSeq2Seq):
    """
    自定义数据整理器，支持混合数据（有的有ref_dist，有的没有）。
    生成 ref_dist_mask 用于在模型 forward 中屏蔽无效 loss。
    """
    def __init__(self, *args, model_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = model_config 

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        
        # 1. 检查是否存在 ref_dist 字段 (即使是 None)
        has_ref_dist_key = "ref_dist" in features[0] if features else False
        ref_dists_raw = None
        
        # 2. 提取 ref_dist 数据 (Pop出来防止干扰父类处理)
        # 注意：这里提取出来的列表可能包含 Numpy数组、List 或者 None
        if has_ref_dist_key:
            ref_dists_raw = [f.pop("ref_dist", None) for f in features]
        
        # 3. 调用父类处理常规字段 (input_ids, labels 等)
        batch = super().__call__(features, return_tensors=return_tensors)
        
        # 4. 处理 ref_dist 和 生成 mask
        if has_ref_dist_key and ref_dists_raw:
            batch_size = len(features)
            
            # --- A. 尝试找到 Batch 中第一个【非 None】的样本来确定维度 ---
            valid_sample = next((x for x in ref_dists_raw if x is not None), None)
            
            if valid_sample is not None:
                # 确定维度
                valid_np = np.array(valid_sample)
                total_dim = valid_np.size
                
                # 获取层数
                if self.model_config is None:
                    raise ValueError("CustomDataCollator 缺少 model_config")
                
                num_layers = getattr(self.model_config, "num_hidden_layers", None)
                if not num_layers: 
                    raise AttributeError("Config 缺少 num_hidden_layers")
                
                # 检查维度合法性
                if total_dim % num_layers != 0:
                    raise ValueError(f"维度错误: Total {total_dim} / Layers {num_layers} 不能整除")
                
                num_experts = total_dim // num_layers

                # --- B. 初始化全 0 张量和 Mask ---
                # Data Tensor: [Batch, Layers, Experts]
                ref_tensor = torch.zeros((batch_size, num_layers, num_experts), dtype=torch.float)
                # Mask Tensor: [Batch, Layers] -> 1.0 表示有效, 0.0 表示无效
                # 这里生成 (Batch, Layers) 是为了方便后续可能的层级加权，或者简单的 (Batch,) 也可以，但层级更通用
                ref_dist_mask = torch.zeros((batch_size, num_layers), dtype=torch.float)

                # --- C. 填充数据 ---
                for i, item in enumerate(ref_dists_raw):
                    if item is not None:
                        # item 是 flat 的 (Layers*Experts)，需要 reshape
                        # item 可能是 list 或 numpy
                        item_arr = np.array(item) 
                        item_tensor = torch.from_numpy(item_arr).float().view(num_layers, num_experts)
                        
                        ref_tensor[i] = item_tensor
                        ref_dist_mask[i] = 1.0 # 标记该样本的所有层为有效

                # --- D. 调整维度并放入 Batch ---
                # 模型通常期望: [Layers, Batch, Experts]
                batch["reference_router_distributions"] = ref_tensor.permute(1, 0, 2)
                
                # Mask 不需要 permute，保持 [Batch, Layers] 传给模型，模型里自己处理转置即可
                batch["ref_dist_mask"] = ref_dist_mask
            
            else:
                # --- E. 如果整个 Batch 全是 None (极其罕见但可能发生) ---
                # 这种情况下不传任何 tensor，模型 forward 需要 handle key 不存在的情况
                pass
                
        return batch

# class CustomDataCollator(DataCollatorForSeq2Seq):
#     """
#     自定义数据整理器，用于处理 ref_dist (参考路由分布).
#     """
#     def __init__(self, *args, model_config=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model_config = model_config # 记得在 main 里实例化时传入 config

#     def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        
#         has_ref_dist = "ref_dist" in features[0] if features else False
#         ref_dists_flat = None
        
#         if has_ref_dist:
#             ref_dists_flat = [f.pop("ref_dist") for f in features]
        
#         batch = super().__call__(features, return_tensors=return_tensors)
        
#         if has_ref_dist and ref_dists_flat:
#             # 1. Numpy -> Tensor
#             ref_numpy = np.array(ref_dists_flat)
#             ref_tensor = torch.from_numpy(ref_numpy).float()
            
#             batch_size, total_dim = ref_tensor.shape
            
#             # =======================================================
#             # 【修正点】: 必须从 config 获取，如果没有 config 则抛出异常
#             # =======================================================
#             if self.model_config is None:
#                 raise ValueError(
#                     "【错误】CustomDataCollator 缺少 model_config。\n"
#                     "请在 main 函数实例化 Trainer 时，向 data_collator 传入 model_config=config 参数。"
#                 )

#             # 自动获取层数 (通常是 num_hidden_layers，部分旧模型是 n_layer)
#             num_layers = getattr(self.model_config, "num_hidden_layers", None)
                
#             if num_layers is None:
#                 raise AttributeError("无法从 model_config 中找到 'num_hidden_layers' 属性，无法确定模型层数。")

#             # 安全检查：确保数据维度是正确的
#             if total_dim % num_layers != 0:
#                 raise ValueError(
#                     f"数据维度错误！\n"
#                     f"数据总长 (Experts*Layers) 为 {total_dim}，但模型层数是 {num_layers}。\n"
#                     f"{total_dim} 无法被 {num_layers} 整除，请检查 np.save 时的形状是否正确。"
#                 )

#             # 自动推算专家数
#             num_experts = total_dim // num_layers
            
#             # =======================================================
            
#             # 还原形状: (B, L*E) -> (B, L, E)
#             ref_tensor = ref_tensor.view(batch_size, num_layers, num_experts)
            
#             # 调整维度: (B, L, E) -> (L, B, E) 以匹配 forward 输入
#             batch["reference_router_distributions"] = ref_tensor.permute(1, 0, 2)
            
#         return batch


def parse_args():
    parser = argparse.ArgumentParser(description="指令微调训练脚本")
    
    # 模型相关参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型的路径或名称")
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用LoRA进行参数高效微调")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA的alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA的dropout率")
    parser.add_argument("--use_4bit", action="store_true",
                        help="是否使用4-bit量化")
    
    # 数据相关参数
    # 在 parse_args() 中替换原来的 dataset_name/train_file 参数
    parser.add_argument("--dataset_list", type=str, nargs="+", required=True,
                        help="多个数据集名称或本地JSON路径(空格分隔)")
    parser.add_argument("--max_samples", type=int, default=None, help="每个数据集筛选的数据条数 (例如 800)")
    parser.add_argument("--eval_split_ratio", type=float, default=0.05,
                        help="当未提供验证文件时，从训练集中划分的比例（默认0.05）")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16,
                        help="数据预处理的工作进程数")
    parser.add_argument("--chat_template", type=str, default=None,
                        help="自定义chat template（可选）")
    
    # 训练相关参数
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="warmup比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="保存策略")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--debug_tiny", action="store_true",
                        help="是否使用微型随机初始化模型进行调试（仅用于跑通流程）")
    
    # DeepSpeed相关参数
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed 自动注入的参数，无需手动设置")

    # JS 散度相关参数
    parser.add_argument("--ref_dist_path_list", type=str, nargs="+", default=None,
                        help="每个数据集对应的路由分布NPY文件路径 (顺序必须与 --dataset_list 对应)")
    parser.add_argument("--js_loss_coef", type=float, default=1.0,  # <-- 2. 新增这一行
                        help="JS散度损失的系数 (将覆盖 config.json 中的值)")
    return parser.parse_args()

def preprocess_function(sample, tokenizer, max_seq_length: int):
    """
    数据预处理函数（带损失掩码）。
    
    1. 生成完整的 input_ids
    2. 生成仅包含 "回答" 部分的 labels，"指令" 部分被设为 -100
    """
    # 1. 解析 messages
    if "messages" in sample:
        messages = sample["messages"]
    else:
        messages = []
        if sample.get("input", ""):
            user_content = f"{sample['instruction']}\n\n{sample['input']}"
        else:
            user_content = sample['instruction']
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": sample['output']})

    # 2. 找出“指令”和“完整对话”的分界点
    #    - "指令"部分：到最后一个 "assistant" 角色之前的所有内容
    #    - "回答"部分：最后一个 "assistant" 角色的 "content"
    
    # 这是一个技巧：我们使用 add_generation_prompt=True 来获取“指令”部分的token
    # 注意：这里我们假设最后一个 message 是 assistant 的回复
    prompt_messages = messages[:-1]
    response_message = messages[-1]

    # 2a. 对“指令”部分进行分词 (不添加eos)
    # add_generation_prompt=True 会在最后添加 "assistant\n" 之类的提示符
    prompt_contents = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        max_length=max_seq_length,
        truncation=True
    )
    
    # 2b. 对“完整对话”进行分词 (添加eos)
    full_contents = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False, # 包含助手的完整回复
        enable_thinking=False,
        max_length=max_seq_length,
        truncation=True,
    )
    tokenized_prompt_contents = tokenizer(prompt_contents,
                                          truncation=True,
                                          max_length=max_seq_length,
                                          padding=False,
                                          return_tensors=None)
    tokenized_prompt_contents["labels"] = tokenized_prompt_contents["input_ids"].copy()
    tokenized_full_contents = tokenizer(full_contents,
                                         truncation=True,
                                         max_length=max_seq_length,
                                         padding=False,
                                         return_tensors=None)

    # print(tokenized_prompt_contents, tokenized_full_contents)

    # 找出 "指令" 的长度
    prompt_length = len(tokenized_prompt_contents['input_ids'])
    tokenized_full_contents["labels"] = [-100] * prompt_length + tokenized_full_contents["input_ids"][prompt_length:]

    # ===== 新增：传递参考路由分布 =====
    if "ref_dist" in sample:
        tokenized_full_contents["ref_dist"] = sample["ref_dist"]
    # ================================

    return tokenized_full_contents


def find_target_modules(model):
    """
    智能查找需要 LoRA 的目标模块。
    
    规则：
    1. 包含: Attention (q,k,v,o), MLP (gate,up,down), MoE Router (gate/router)
    2. 排除: embed_tokens, lm_head, norm, buffers
    """
    # === 1. 白名单：只允许这些后缀的层被选中 ===
    target_whitelist = [
        # --- Attention (注意力) ---
        "q_proj", "k_proj", "v_proj", "o_proj", 
        
        # --- MLP (前馈网络) ---
        "gate_proj", "up_proj", "down_proj", 
        
        # --- MoE Gate/Router (路由) ---
        "gate",   # DeepSeek, Mixtral, Llama-MoE 通常叫 gate
        "router", # 部分其他变体
        "w_gate", # 某些旧实现
    ]

    # === 2. 黑名单：显式禁止 (双重保险) ===
    # 注意：embed_tokens 通常是 nn.Embedding，本来就不会被 isinstance(Linear) 抓到，
    # 但这里写上是为了逻辑上的绝对明确。
    exclude_blacklist = [
        "embed_tokens",   # 输入嵌入层
        "lm_head",        # 输出层
        "norm", "ln_f",   # 归一化层
        "score",          # 某些特定的评分层
    ]
    
    lora_module_names = set()

    for name, module in model.named_modules():
        # 只处理线性层 (Linear)
        if isinstance(module, torch.nn.Linear):
            # 提取层名称的最后一部分，例如 'model.layers.0.self_attn.q_proj' -> 'q_proj'
            module_suffix = name.split('.')[-1]
            
            # 核心判定逻辑：必须在白名单中，且不在黑名单中
            if module_suffix in target_whitelist and module_suffix not in exclude_blacklist:
                lora_module_names.add(module_suffix)
    
    return list(lora_module_names)


def main():
    args = parse_args()
    set_seed(args.seed)

    # ===== 新增：检查路由分布文件 =====
    use_ref_dist = args.ref_dist_path_list is not None
    if use_ref_dist:
        if len(args.ref_dist_path_list) != len(args.dataset_list):
            # raise ValueError(
            #     f"[错误] --ref_dist_path_list (数量 {len(args.ref_dist_path_list)}) "
            #     f"必须与 --dataset_list (数量 {len(args.dataset_list)}) 一一对应。"
            # )
            print(f"[Warning] --ref_dist_path_list (数量 {len(args.ref_dist_path_list)}) "
                  f"与 --dataset_list (数量 {len(args.dataset_list)}) 不对应。"
                )
        print(f"检测到 {len(args.ref_dist_path_list)} 个路由分布文件。将计算 JS 散度损失。")
    # ================================

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True
    )
    tokenizer.padding_side = "left" # 对于Causal LM微调，设置为left更安全
    # 应用自定义chat template（如果提供）
    if args.chat_template:
        tokenizer.chat_template = args.chat_template
    
    # 打印使用的chat template信息
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"使用的chat template: {tokenizer.chat_template[:100]}...")
    else:
        print("未找到chat template，将使用默认格式")
    
    # ===== 加载多个数据集并划分验证集 =====
    train_list = []
    val_list = []

    def load_clean_jsonl(path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("messages"), list):
                    records.append(obj)
        return DatasetDict({"train": Dataset.from_list(records)})

    for i, ds_path in enumerate(args.dataset_list):
        suffix = Path(ds_path).suffix.lower()
        if suffix in [".json", ".jsonl"]:
            # ds = load_dataset("json", data_files={"train": ds_path})
            ds = load_clean_jsonl(ds_path)
            # ----- 新增：加载并添加路由分布 -----
            if use_ref_dist:
                try:
                    ref_dist_path = args.ref_dist_path_list[i]
                    if not os.path.exists(ref_dist_path):
                        raise FileNotFoundError(f"找不到路由分布文件: {ref_dist_path} (对应 {ds_path})")
                    
                    print(f"正在加载 {ds_path} 对应的路由分布 {ref_dist_path}...")
                    
                    # 加载 NPY 文件，形状 (num_samples, num_layers, num_experts)
                    ref_dist_data = np.load(ref_dist_path)
                    
                    # 验证样本数量是否匹配
                    if len(ds["train"]) != len(ref_dist_data):
                        raise ValueError(
                            f"数据集 {ds_path} (样本数 {len(ds['train'])}) 与 "
                            f"路由分布 {ref_dist_path} (样本数 {len(ref_dist_data)}) 的数量不匹配。"
                        )
                    
                    N = ref_dist_data.shape[0]
                    ref_dist_flat = ref_dist_data.reshape(N, -1)
                    
                    # 添加为新列。注意：.add_column 需要一个 list
                    # ds["train"] = ds["train"].add_column("ref_dist", ref_dist_data.tolist())
                    ds["train"] = ds["train"].add_column("ref_dist", list(ref_dist_flat))
                except Exception as e:
                    print(f"[Warning] 数据集 {ds_path} 不使用路由分布: {e}")

                    # 【关键修改】：如果报错或没有文件，填充 None
                    # 必须保证列存在，concatenate_datasets 才不会报错
                    none_column = [None] * len(ds["train"])
                    ds["train"] = ds["train"].add_column("ref_dist", none_column)

            if args.max_samples is not None:
                print(f"筛选数据集 {ds_path} : 取前 {args.max_samples} 条 (Seed: {args.seed})")
                limit = min(len(ds["train"]), args.max_samples)
                ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(limit))
            # ---------------------------------
            split_data = ds["train"].train_test_split(
                test_size=args.eval_split_ratio,
                seed=args.seed,
                shuffle=True
            )
            train_list.append(split_data["train"])
            val_list.append(split_data["test"])
        else:
            ds = load_dataset(ds_path)
            if use_ref_dist:
                print(f"[警告] 暂不支持为 Hub 数据集 {ds_path} 加载路由分布，将跳过 JS 损失。")

            # 兼容 Hub 数据集的筛选
            if args.max_samples is not None:
                 limit = min(len(ds["train"]), args.max_samples)
                 ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(limit))

            if "validation" in ds:
                train_list.append(ds["train"])
                val_list.append(ds["validation"])
            else:
                split_data = ds["train"].train_test_split(
                    test_size=args.eval_split_ratio,
                    seed=args.seed,
                    shuffle=True
                )
                train_list.append(split_data["train"])
                val_list.append(split_data["test"])

    train_data_raw = concatenate_datasets(train_list)
    val_data_raw = concatenate_datasets(val_list) if val_list else None

    train_data = train_data_raw.shuffle(seed=args.seed).map(
        partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length),
        batched=False,
        num_proc=args.preprocessing_num_workers,
        remove_columns=train_data_raw.column_names
    )

    val_data = None
    if val_data_raw:
        val_data = val_data_raw.shuffle(seed=args.seed).map(
            partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length),
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=val_data_raw.column_names
        )

    print("正在加载 config...")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    config.use_cache = False # 节省显存
    config.output_router_logits = True

    if use_ref_dist:
        config.js_loss_coef = args.js_loss_coef
        print(f"✅ 已将 JS Loss 系数 ({config.js_loss_coef}) 注入模型 config。")
    else:
        # 如果不使用 ref_dist，将系数设为 0.0，
        # 这样模型加载时 config 中有值，但计算JS loss时结果为 0。
        config.js_loss_coef = 0.0
    # 加载模型
    model_kwargs = {
        "dtype": torch.bfloat16,
    }
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        **model_kwargs
    )

    # 如果使用4-bit量化，准备模型
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 如果使用LoRA
    if args.use_lora:
        target_modules = find_target_modules(model)
        print(target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print("="*50)
        model.print_trainable_parameters()
        print("="*50)
    # 新增：如果不使用 LoRA，手动计算并打印参数
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("="*50)
        print(f"参数统计 (非LoRA, 选择性微调):")
        print(f"  - 总参数 (Total): {total_params:.2f} M")
        print(f"  - 可训练参数 (Trainable): {trainable_params/1e6:.2f} M")
        if total_params > 0:
            print(f"  - 可训练比例 (Ratio): {100 * trainable_params / total_params:.4f}%")
        print("="*50)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if val_data else "no",
        eval_steps=args.eval_steps if val_data else None,
        metric_for_best_model="loss" if val_data else None,
        greater_is_better=False if val_data else None,
        deepspeed=args.deepspeed_config,
        fp16=True if not torch.cuda.is_bf16_supported() else False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        gradient_checkpointing=True,
        report_to="wandb",
        seed=args.seed,
        dataloader_num_workers=4,      # <--- 新增：加速数据读取
        remove_unused_columns=False,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator = CustomDataCollator(
            tokenizer,
            model_config=config,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存最终模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 如果使用LoRA，单独保存adapter
    if args.use_lora:
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))


if __name__ == "__main__":
    main()
