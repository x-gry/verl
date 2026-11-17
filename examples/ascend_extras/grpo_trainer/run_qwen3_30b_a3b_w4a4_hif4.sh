#!/usr/bin/env bash
# Single-node Qwen3-30B-A3B GRPO: FSDP Turbo MXFP8 training, W4A4_HIFP4 rollout.
# Set MODEL_PATH, TRAIN_FILE and TEST_FILE before running from the veRL root.
# The model's quant_model_description.json must select W4A4_HIFP4.
set -euo pipefail

start_time="${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
project_name="${VERL_PROJECT_NAME:-GRPO-Qwen3-30B}"

nnodes=1
npus_per_node=8
visible_devices_default=0,1,2,3,4,5,6,7
turbo_ep_size=8
turbo_efsdp_size=1
turbo_fsdp_size=8
# TP=4 and DP=2 use eight NPUs; rollout EP equals TP * DP.
rollout_tp_size=4
rollout_dp_size=2
rollout_ep_size=8
max_response_length="${VERL_MAX_RESPONSE_LENGTH:-512}"
rollout_max_num_seqs="${VERL_VLLM_MAX_NUM_SEQS:-8}"
rollout_max_num_batched_tokens="${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
train_batch_size="${VERL_TRAIN_BATCH_SIZE:-16}"
rollout_n="${VERL_ROLLOUT_N:-4}"
ppo_mini_batch_size="${VERL_PPO_MINI_BATCH_SIZE:-${train_batch_size}}"
actor_moe_chunk_tokens="${VERL_ACTOR_MOE_CHUNK_TOKENS:-512}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the Qwen3-30B-A3B model directory}"
exp_name="${VERL_EXPERIMENT_NAME:-qwen3_30b_w4a4_hif4_${start_time}}"
save_freq_default=10
profiler_log_level_default=WARN

save_freq="${VERL_SAVE_FREQ:-${save_freq_default}}"
rollout_enforce_eager="${VERL_VLLM_ENFORCE_EAGER:-True}"
resume_mode="${VERL_RESUME_MODE:-disable}"
resume_from_path="${VERL_RESUME_FROM_PATH:-}"
train_file="${TRAIN_FILE:?Set TRAIN_FILE to the GSM8K training parquet file}"
test_file="${TEST_FILE:?Set TEST_FILE to the GSM8K validation parquet file}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices_default}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NON_MEGATRON="${NON_MEGATRON:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-${profiler_log_level_default}}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM="${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-50000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17340}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export P2P_HCCL_BUFFSIZE="${P2P_HCCL_BUFFSIZE:-20}"

local_ray_owned=0
cleanup_local_ray() {
    if (( local_ray_owned == 1 )); then
        echo "Stopping local Ray runtime"
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_local_ray EXIT INT TERM

# Start a single-node Ray runtime; CPU and object-store sizing use Ray defaults.
node_ip="${NODE_IP:-127.0.0.1}"
ray_args=(
    --head
    --port="${RAY_PORT:-6379}"
    --node-ip-address="${node_ip}"
    --dashboard-host="${node_ip}"
    --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"
    --resources="{\"NPU\": ${npus_per_node}}"
)
if [[ -n "${RAY_NUM_CPUS:-}" ]]; then
    ray_args+=(--num-cpus="${RAY_NUM_CPUS}")
fi
if [[ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]]; then
    ray_args+=(--object-store-memory="${RAY_OBJECT_STORE_MEMORY}")
fi
ray start "${ray_args[@]}"
local_ray_owned=1

max_prompt_length=1024
rollout_max_model_len="${VERL_VLLM_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}"
ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))

RAY_CONFIG=(
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_RT_VISIBLE_DEVICES=\"${ASCEND_RT_VISIBLE_DEVICES}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS=\"${OMP_NUM_THREADS}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.NON_MEGATRON=\"${NON_MEGATRON}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED=\"${PYTHONUNBUFFERED}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED=\"${PYTHONHASHSEED}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM=\"${TOKENIZERS_PARALLELISM}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=\"${VLLM_LOGGING_LEVEL}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=\"${VERL_LOGGING_LEVEL}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DISABLE_COMPILE_CACHE=\"${VLLM_DISABLE_COMPILE_CACHE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BATCH_INVARIANT=\"${VLLM_BATCH_INVARIANT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ALLOW_RUNTIME_LORA_UPDATING=\"${VLLM_ALLOW_RUNTIME_LORA_UPDATING}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1=\"${VLLM_USE_V1}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_FLASHCOMM=\"${VLLM_ASCEND_ENABLE_FLASHCOMM}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ=\"${VLLM_ASCEND_ENABLE_NZ}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=\"${CUDA_DEVICE_MAX_CONNECTIONS}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.MULTI_STREAM_MEMORY_REUSE=\"${MULTI_STREAM_MEMORY_REUSE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.CPU_AFFINITY_CONF=\"${CPU_AFFINITY_CONF}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_OP_EXPANSION_MODE=\"${HCCL_OP_EXPANSION_MODE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE=\"${HCCL_HOST_SOCKET_PORT_RANGE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE=\"${HCCL_NPU_SOCKET_PORT_RANGE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_IF_BASE_PORT=\"${HCCL_IF_BASE_PORT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_EXEC_TIMEOUT=\"${HCCL_EXEC_TIMEOUT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_CONNECT_TIMEOUT=\"${HCCL_CONNECT_TIMEOUT}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_BUFFSIZE=\"${HCCL_BUFFSIZE}\""
    "++ray_kwargs.ray_init.runtime_env.env_vars.P2P_HCCL_BUFFSIZE=\"${P2P_HCCL_BUFFSIZE}\""
)

DATA_CONFIG=(
    data.train_files="${train_file}"
    data.val_files="${test_file}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=False
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${npus_per_node}
    trainer.nnodes=${nnodes}
    trainer.balance_batch=False
    trainer.resume_from_path="${resume_from_path}"
    trainer.resume_mode=${resume_mode}
    trainer.save_freq=${save_freq}
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_epochs=10
)

MODEL_CONFIG=(
    "actor_rollout_ref.model.path=${model_path}"
    actor_rollout_ref.model.trust_remote_code=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len}
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}
    actor_rollout_ref.rollout.data_parallel_size=${rollout_dp_size}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=true
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep_size}
    actor_rollout_ref.rollout.name=vllm
    # Select the Ascend schemes described in the model quantization config.
    actor_rollout_ref.rollout.quantization=ascend
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    # Actor weight synchronization populates rollout parameters before generation.
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

case "${rollout_enforce_eager,,}" in
    false|0|no|off)
        ROLLOUT_CONFIG+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16]'
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=16
        )
        ;;
esac

FSDP_CONFIG=(
    actor_rollout_ref.actor.strategy=fsdp_turbo
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    +actor_rollout_ref.actor.fsdp_config.turbo_config.model.attn_implementation=eager
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]
    '++actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules={model.layers.\{*\}:{},model.embed_tokens:{},lm_head:{}}'
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=1
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.hook_modules=["model.layers.{*}"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tp_plan.colwise_parallel=["*.q_proj","*.k_proj","*.v_proj"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.tp_plan.rowwise_parallel=["*.o_proj"]'
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=${turbo_fsdp_size}
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.expert_parallel_size=${turbo_ep_size}
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches=[{target_functions:['transformers.models.qwen3_moe.modeling_qwen3_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=["model.layers.{*}.mlp.experts"]'
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=all2all
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=${turbo_efsdp_size}
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=["model.layers.{*}.mlp.experts"]'
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]'
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.ref.use_torch_compile=False
)

FSDP_CONFIG+=(
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.chunk_batch=False
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.chunk_batch_plan={chunk_mbs:${actor_moe_chunk_tokens},apply_modules:['model.layers.{*}.mlp.experts'],batch_dim:1,chunk_arg_indexs:[0],chunk_kwarg_names:[]}"
)

# Quantize decoder Linear and MoE computation to MXFP8 E4M3.
FSDP_CONFIG+=(
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_recipe=mxfp8"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_format=E4M3"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.converters=['quantize.linear.mx','quantize.moe.mx']"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_apply_modules=['model.layers.{*}']"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=True"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.quant_gmm=true"
    "+actor_rollout_ref.actor.fsdp_config.turbo_config.quantization.quantization_plan.gemm_gradient_accumulation_fusion=false"
)

REF_CONFIG=(
    actor_rollout_ref.ref.strategy=fsdp_turbo
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16
    +actor_rollout_ref.ref.fsdp_config.turbo_config.model.attn_implementation=eager
    +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]
    '++actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules={model.layers.\{*\}:{},model.embed_tokens:{},lm_head:{}}'
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=0
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=0
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fsdp_plan.hook_modules=["model.layers.{*}"]'
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.tp_plan.colwise_parallel=["*.q_proj","*.k_proj","*.v_proj"]'
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.tp_plan.rowwise_parallel=["*.o_proj"]'
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=${turbo_fsdp_size}
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.expert_parallel_size=${turbo_ep_size}
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches=[{target_functions:['transformers.models.qwen3_moe.modeling_qwen3_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=["model.layers.{*}.mlp.experts"]'
    +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=all2all
    +actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=${turbo_efsdp_size}
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=["model.layers.{*}.mlp.experts"]'
    '+actor_rollout_ref.ref.fsdp_config.turbo_config.memory.recompute_plan=["model.layers.{*}"]'
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len}
    # Use the same MXFP8 recipe for reference log-probability computation.
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_recipe=mxfp8"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_format=E4M3"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.converters=['quantize.linear.mx','quantize.moe.mx']"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_apply_modules=['model.layers.{*}']"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=True"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.quant_gmm=true"
    "+actor_rollout_ref.ref.fsdp_config.turbo_config.quantization.quantization_plan.gemm_gradient_accumulation_fusion=false"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=dp \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${FSDP_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${RAY_CONFIG[@]}" \
    "$@" 2>&1 | tee "${VERL_LOG_FILE:-run_qwen3_30b_w4a4_hif4.log}"
