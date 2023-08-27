# ------------------------------------------LSB-----------------------------------------------------------

import torch


def post_embed_watermark(embedding, watermark, mode, start_pos=900, perturbation=10):
    """
    Embed the watermark into the embedding using differnt post-processing strategy.

    Args:
    - embedding (torch.Tensor): The tensor of face embeddings, shape [batch_size, embedding_size].
    - watermark (torch.Tensor): The tensor of binary watermarks, shape [batch_size, watermark_size].
    - mode (str): The post-processing strategy to use. One of ['LSB', 'FTT', 'Noise'].

    Returns:
    - torch.Tensor: Watermarked embeddings.
    """
    
    if mode == "LSB":
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
    
    elif mode == "FTT":
        
        watermark_size = watermark.shape[1]
        freq_repr = torch.fft.fft(embedding)
        freq_repr[:, start_pos : start_pos + watermark_size] = watermark.to(torch.complex64)
        embedding_restore = torch.fft.ifft(freq_repr)
        
        return embedding_restore

    elif mode == "Noise":
        # perturbation = 0.01
        
        watermark = watermark * 2 -1
        
        return embedding + perturbation * watermark


    else:
        raise ValueError("Invalid post-processing mode: {mode}, which must be one of ['LSB', 'FTT', 'Noise'].")


        


def post_extract_watermark(watermarked_embedding, watermark_size, mode, start=900):
    """
    Extract the watermark from the watermarked embedding using LSB strategy.

    Args:
    - embedding (torch.Tensor): The tensor of watermarked embeddings, shape [batch_size, embedding_size].
    - watermark_size (int): The size of the watermark to extract.
    - mode (str): The post-processing strategy to use. One of ['LSB', 'FTT', 'Noise'].


    Returns:
    - torch.Tensor: Extracted binary watermark.
    """
    if mode == "LSB":
        watermarked_embedding = watermarked_embedding * (10**5)
        watermarked_embedding = torch.round(watermarked_embedding)
        extracted_watermark = watermarked_embedding % 2
        return extracted_watermark
    elif mode == "FTT":

        freq_repr = torch.fft.fft(watermarked_embedding)
        
        watermark_restore = freq_repr[:, start:start+watermark_size].to(torch.float32)
        watermark_restore = (watermark_restore > 0.5).to(torch.int64)

        return watermark_restore
    
    elif mode == "Noise":
        # 阈值，用于确定位是0还是1
        threshold = watermarked_embedding.mean()
        return (watermarked_embedding > threshold).to(torch.int64)
        
    else:
        raise ValueError("Invalid post-processing mode: {mode}, which must be one of ['LSB', 'FTT', 'Noise'].")




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
# 

#---------------------------------------Spread Spectrum Watermarking-------------------------

import torch
import torch.nn.functional as F

def generate_pseudo_random_sequence(embedding_size, seed):
    torch.manual_seed(seed) # 为了保证每次生成相同的序列
    return torch.randn(embedding_size)

def spread_spectrum_embedding(embedding, watermark, seed, strength=0.1):
    """Embed watermark into embedding using spread spectrum."""
    sequence = generate_pseudo_random_sequence(len(embedding), seed)
    watermarked_embedding = embedding + strength * sequence * watermark
    return watermarked_embedding

def spread_spectrum_extraction(watermarked_embedding, original_embedding, seed, strength=0.1):
    """Extract watermark from watermarked embedding using spread spectrum."""
    sequence = generate_pseudo_random_sequence(len(watermarked_embedding), seed)
    diff = watermarked_embedding - original_embedding
    watermark = torch.sum(diff * sequence) / (strength * len(embedding))
    return torch.round(F.sigmoid(watermark))  # Assuming binary watermark

# 示例
embedding_size = 1024
seed = 42

embedding = torch.randn(embedding_size)
watermark = torch.tensor(1.0)  # Here, a single bit watermark is used, but you can extend this

# 嵌入水印
watermarked_embedding = spread_spectrum_embedding(embedding, watermark, seed)

# 提取水印
extracted_watermark = spread_spectrum_extraction(watermarked_embedding, embedding, seed)

# print(f"Original Watermark: {watermark}")
# print(f"Extracted Watermark: {extracted_watermark}")



#-------------------------------Frequency-domain Embedding-----------------------------
import numpy as np
import torch

def embed_watermark_in_frequency(embedding, watermark):
    # 快速傅立叶变换
    freq_repr = np.fft.fft(embedding.numpy())
    
    # 将水印嵌入到某些选定的频率分量中
    freq_repr[10:10+len(watermark)] += watermark.numpy()
    
    # 逆FFT转换回时域
    watermarked_embedding = np.fft.ifft(freq_repr)
    
    return torch.tensor(watermarked_embedding, dtype=torch.float32)

def extract_watermark_from_frequency(watermarked_embedding, length):
    freq_repr = np.fft.fft(watermarked_embedding.numpy())
    
    # 从相同的位置提取水印
    extracted_watermark = freq_repr[10:10+length]
    
    return torch.tensor(extracted_watermark, dtype=torch.float32)


#------------------------perturbation embedding---------------------------
def embed_watermark_with_perturbation(embedding, watermark):
    # 扰动的大小
    perturbation = 0.01
    
    # 基于水印信息增加扰动
    watermarked_embedding = embedding.clone()
    for i, bit in enumerate(watermark):
        if bit == 1:
            watermarked_embedding[i] += perturbation
        else:
            watermarked_embedding[i] -= perturbation

    return watermarked_embedding

def extract_watermark_with_perturbation(watermarked_embedding, length):
    # 阈值，用于确定位是0还是1
    threshold = watermarked_embedding.mean().item()
    
    extracted_watermark = []
    for i in range(length):
        bit = 1 if watermarked_embedding[i].item() > threshold else 0
        extracted_watermark.append(bit)

    return torch.tensor(extracted_watermark, dtype=torch.int)





if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    # Test the post_processing watermark embedding and extraction
    embedding = torch.randn(3, 8) * 10  # Random embeddings
    watermark = torch.randint(0, 2, (3, 8))  # Random binary watermarks of size 32
    watermarked_embedding = post_embed_watermark(embedding, watermark, mode="LSB")
    extracted_watermark = post_extract_watermark(watermarked_embedding, 8, mode="LSB")
    print((extracted_watermark == watermark).all())
    # Check if extracted watermark is same as original watermark
