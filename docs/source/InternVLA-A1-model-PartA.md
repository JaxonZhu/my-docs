# InternVLA-A1 源码分析：2B 模型的联合注意力

这篇源码笔记沿着 [InternVLA-A1 2B](InternVLA-A1-paper.md) 的参数配置、前向计算和联合注意力展开。我主要关注理解、生成与动作三个专家如何交换信息，以及联合计算后的结果如何回到各自的分支。

下图是整体流程：各专家生成 Q、K、V，沿序列维度拼接后计算注意力，再将结果拆回各专家，继续输出投影、残差连接与 MLP 计算。

![InternVLA-A1 联合注意力总览：三个专家的 Q、K、V 拼接计算注意力，再按序列长度拆分回各专家分支](images/InternVLA-A1/joint_attention_total_figure.png)

## 配置说明

> 所有配置参数以 GitHub 上开源呈现为主。

> "Specifically, [InternVLA-A1 (2B)](InternVLA-A1-paper.md) utilizes InternVL3-1B as the understanding expert. Its generative and action experts are derived from the transformer blocks of Qwen2.5 — the underlying LLM of InternVL3."

## 模型参数配置

`InternVLForConditionalGeneration.from_pretrained()` 加载预训练权重：模型参数从 Hugging Face Hub 下载的文件 `OpenGVLab/InternVL3-1B-pt` 中加载；结构由预训练模型的配置文件决定，虽然此处传入了 `config=vlm_config_hf` ，但通常需匹配预训练权重。

`Qwen2ForCausalLM(config=gen_expert_config_hf)` **随机初始化**：模型根据提供的 `config=gen_expert_config_hf` 构建结构，但所有参数（权重、偏置）都是**随机生成的数值**。作为一个全新的网络结构，需要在后续的训练步骤中从头开始学习。

![模型参数配置0](images/InternVLA-A1/model_config_0.png)

![模型参数配置1](images/InternVLA-A1/model_config_1.png)

![模型参数配置2](images/InternVLA-A1/model_config_2.png)

![模型参数配置3](images/InternVLA-A1/model_config_3.png)

![模型参数配置4](images/InternVLA-A1/model_config_4.png)

## 模型前向计算

`embed_image()` 和 `embed_language_tokens()` 方法最终都是将各种模态信息拆解成 `[批维度, 序列维度, 隐藏层维度]` 统一，以便后续联合 Transformers 进行处理.

![前向计算0](images/InternVLA-A1/model_expert_forward_0.png)

![前向计算1](images/InternVLA-A1/model_expert_forward_1.png)

推理时，运行 `gen_expert` 模型会接收来自 `und_expert` 专家前向计算的 KV 缓存，在 `.forward()` 方法内部会合并 `und_expert` 和自身前向计算的 KV 缓存。在后续推理时，`attention_mask` 会兼并 `und_expert` 的掩码，同时 `position_ids` 也是从 `und_expert` 产生的结果上继续累加。

![前向计算2](images/InternVLA-A1/model_expert_forward_2.png)

![前向计算3](images/InternVLA-A1/model_expert_forward_3.png)

![前向计算4](images/InternVLA-A1/model_expert_forward_4.png)

![前向计算5](images/InternVLA-A1/model_expert_forward_5.png)

## 联合注意力核心

下面的截图对应 Q、K、V 的联合计算；后面的代码展示如何按各专家的序列长度拆分注意力输出，并送回各自的输出层与 MLP。

![联合注意力核心截图 1](images/InternVLA-A1/joint_attention_core_0.png)

![联合注意力核心截图 2](images/InternVLA-A1/joint_attention_core_1.png)

![联合注意力核心截图 3](images/InternVLA-A1/joint_attention_core_2.png)

```python
# 根据 seq_len * 3 长度分段提取不同注意力结果, 输入至不同的 expert's output / mlp 层.
outputs_embeds = []
start_pos = 0

for i, hidden_states in enumerate(inputs_embeds):

    # 取出当前专家的 Transformer 层
    # 找到起始位置和结束位置并对注意力结果进行截断
    # 将截断后的注意力结果通过当前专家的输出投影层
    # 最后进行残差计算
    # out_emb: [batch_size, seq_len, hidden_size]
    layer = models[i].layers[layer_idx]
    end_pos = start_pos + hidden_states.shape[1]
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
    out_emb = out_emb + hidden_states

    # 进行 post-attention + mlp 计算
    # 最后再次进行残差计算
    after_first_residual = out_emb.clone()
    out_emb = layer.post_attention_layernorm(out_emb)
    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        out_emb = out_emb.to(dtype=torch.bfloat16)
    out_emb = layer.mlp(out_emb)
    out_emb = out_emb + after_first_residual

    # 将结果添加到输出列表, 并更新起始位置开始下一个专家的处理
    outputs_embeds.append(out_emb)
    start_pos = end_pos
```
