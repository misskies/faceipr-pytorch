# ------------------------------------------LSB-----------------------------------------------------------

import torch


def embed_watermark(embedding, watermark):
    """
    Embed the watermark into the embedding using LSB strategy.

    Args:
    - embedding (torch.Tensor): The tensor of face embeddings, shape [batch_size, embedding_size].
    - watermark (torch.Tensor): The tensor of binary watermarks, shape [batch_size, watermark_size].

    Returns:
    - torch.Tensor: Watermarked embeddings.
    """
    # Ensure the watermark fits within the embedding
    assert embedding.shape[1] >= watermark.shape[1], "Watermark size is larger than embedding size."
    embedding = embedding * (10**5)
    # Zero out the least significant bit
    embedding = embedding.to(int)
    embedding = (embedding // 2) * 2
    # Add the watermark bits to the embedding
    watermarked_embedding = embedding + watermark
    watermarked_embedding = watermarked_embedding / (10**5)

    return watermarked_embedding


def extract_watermark(embedding, watermark_size):
    """
    Extract the watermark from the watermarked embedding using LSB strategy.

    Args:
    - embedding (torch.Tensor): The tensor of watermarked embeddings, shape [batch_size, embedding_size].
    - watermark_size (int): The size of the watermark to extract.

    Returns:
    - torch.Tensor: Extracted binary watermark.
    """
    embedding = embedding * (10**5)
    embedding = torch.round(embedding)
    extracted_watermark = embedding % 2
    return extracted_watermark

if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    # Test the LSB watermark embedding and extraction
    embedding = torch.randn(3, 8) * 10  # Random embeddings
    watermark = torch.randint(0, 2, (3, 8))  # Random binary watermarks of size 32
    watermarked_embedding = embed_watermark(embedding, watermark)
    extracted_watermark = extract_watermark(watermarked_embedding, 8)
    print((extracted_watermark == watermark).all())
    # Check if extracted watermark is same as original watermark

# -------------------------------- Loss function baseline----------------------------------------

## 水印嵌入

# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# # 定义一个简化的FaceNet模型
# class SimpleFaceNet(nn.Module):
#     def __init__(self):
#         super(SimpleFaceNet, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 224 * 224, 128),
#             nn.Tanh()  # 使输出在[-1,1]范围内
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# # 定义三元组损失
# triplet_loss = nn.TripletMarginLoss(margin=1.0)
#
# # 实例化模型和优化器
# model = SimpleFaceNet()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 假设的训练数据
# anchor = torch.randn(8, 3, 224, 224)
# positive = torch.randn(8, 3, 224, 224)
# negative = torch.randn(8, 3, 224, 224)
#
# # 假设的二进制水印向量
# binary_watermark = torch.randint(0, 2, (8, 128)) * 2 - 1  # 转换为-1和1
#
# # 前向传播
# anchor_embedding = model(anchor)
# positive_embedding = model(positive)
# negative_embedding = model(negative)
#
# # 计算原始三元组损失
# loss_triplet = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
#
# # 计算与水印的损失
# loss_watermark = nn.MSELoss()(anchor_embedding, binary_watermark)
#
# # 结合两个损失
# lambda_watermark = 0.1
# combined_loss = loss_triplet + lambda_watermark * loss_watermark
#
# # 反向传播和优化
# optimizer.zero_grad()
# combined_loss.backward()
# optimizer.step()
#

# 在这个示例中，我们首先使用了nn.Tanh()激活函数来确保模型输出值在[-1,1]的范围内。然后，我们创建了一个在-1和1之间的水印，这样就可以使用均方误差与embedding进行比较。
# 这种方法确保了水印信息会被嵌入到embedding中，而不会对人脸识别的主要任务产生太大的干扰。您可能需要根据实际需求和实验结果调整lambda_watermark的值

# 水印长度不一致，可以采取：重复水印或者嵌入embedding特定区域
##重复水印

def fit_watermark_to_embedding(watermark, embedding_size):
    """
    Adjust the watermark to fit the size of the embedding by repeating or truncating.

    Args:
    - watermark (torch.Tensor): The tensor of binary watermarks, shape [batch_size, watermark_size].
    - embedding_size(int): The size of the embedding.

    Returns:
    - torch.Tensor: Adjusted watermark with shape [batch_size, embedding_size].
    """
    adjusted_watermark = []

    for wm in watermark:
        repeated_wm = wm.repeat((embedding_size // wm.size(0)) + 1)[:embedding_size]
        adjusted_watermark.append(repeated_wm)

    return torch.stack(adjusted_watermark)


# 示例
# embedding_size = 128
# watermark = torch.randint(0, 2, (8, 32)) * 2 - 1  # 假设的二进制水印向量
# adjusted_watermark = fit_watermark_to_embedding(watermark, embedding_size)
#
# print(adjusted_watermark.shape)  # torch.Size([8, 128])


## 水印提取

def extract_binary_watermark(embedding):
    """
    Extract binary watermark from the embedding.

    Args:
    - embedding (torch.Tensor): The tensor of watermarked embeddings, shape [batch_size, watermark_size].

    Returns:
    - torch.Tensor: Extracted binary watermark.
    """
    binary_watermark = (embedding.sign() + 1) // 2  # Convert -1 to 0 and 1 to 1
    return binary_watermark


# 示例
# watermarked_embedding = model(anchor)
# extracted_watermark = extract_binary_watermark(watermarked_embedding)
#
# print(extracted_watermark)

# extract_binary_watermark函数首先计算embedding的符号（sign），这会将所有负值转换为-1，所有正值转换为1。接下来，我们将结果加1并除以2，以将-1转换为0，1保持为1。这样我们就获得了原始的二进制水印