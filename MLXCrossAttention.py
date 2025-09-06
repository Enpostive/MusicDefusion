import mlx.core as mx
import mlx.nn as nn
import mlx.nn.init as init
import math

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for MLX.
    
    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_bias: Whether to use bias in linear projections
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, use_bias: bool = True):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Optional feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=use_bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model, bias=use_bias),
            nn.Dropout(dropout)
        )

    def init_weights(self, init_fn = init.glorot_uniform()):
        """Initialize weights using Xavier/Glorot initialization."""
        # Initialize linear projections with Xavier uniform
        self.q_proj.weight = init_fn(self.q_proj.weight)
        self.k_proj.weight = init_fn(self.k_proj.weight)
        self.v_proj.weight = init_fn(self.v_proj.weight)
        self.out_proj.weight = init_fn(self.out_proj.weight)

        # Initialize biases to zero if they exist
        if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
            self.q_proj.bias = mx.zeros_like(self.q_proj.bias)
        if hasattr(self.k_proj, 'bias') and self.k_proj.bias is not None:
            self.k_proj.bias = mx.zeros_like(self.k_proj.bias)
        if hasattr(self.v_proj, 'bias') and self.v_proj.bias is not None:
            self.v_proj.bias = mx.zeros_like(self.v_proj.bias)
        if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
            self.out_proj.bias = mx.zeros_like(self.out_proj.bias)
        
        # Initialize FFN layers
        for layer in self.ffn.layers:
            if isinstance(layer, nn.Linear):
                layer.weight = init_fn(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias = mx.zeros_like(layer.bias)
        
        # Layer norm weights are typically initialized to 1, bias to 0
        self.norm1.weight = mx.ones_like(self.norm1.weight)
        self.norm1.bias = mx.zeros_like(self.norm1.bias)
        self.norm2.weight = mx.ones_like(self.norm2.weight)
        self.norm2.bias = mx.zeros_like(self.norm2.bias)
    
    def __call__(self, query, key_value, mask=None):
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key_value: Key-Value tensor [batch_size, seq_len_kv, d_model]
            mask: Optional attention mask [batch_size, seq_len_q, seq_len_kv]
        
        Returns:
            Output tensor [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]
        
        # Store residual connection
        residual = query
        
        # Layer normalization (pre-norm)
        query = self.norm1(query)
        key_value = self.norm2(key_value)
        
        # Linear projections
        Q = self.q_proj(query)    # [batch_size, seq_len_q, d_model]
        K = self.k_proj(key_value)  # [batch_size, seq_len_kv, d_model]
        V = self.v_proj(key_value)  # [batch_size, seq_len_kv, d_model]
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.n_heads, self.d_head)
        K = K.reshape(batch_size, seq_len_kv, self.n_heads, self.d_head)
        V = V.reshape(batch_size, seq_len_kv, self.n_heads, self.d_head)
        
        # Transpose to [batch_size, n_heads, seq_len, d_head]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mx.expand_dims(mask, axis=1)  # [batch_size, 1, seq_len_q, seq_len_kv]
            scores = mx.where(mask, scores, mx.full_like(scores, -float('inf')))
        
        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, V)
        
        # Reshape back to [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len_q, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # First residual connection
        output = output + residual
        
        # Feed-forward network with second residual connection
        ffn_output = self.ffn(output)
        output = output + ffn_output
        
        return output


class SimpleCrossAttention(nn.Module):
    """
    Simplified cross-attention without FFN for more focused usage.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def init_weights(self, init_fn = init.glorot_uniform()):
        """Initialize weights using Xavier/Glorot initialization."""

        # Initialize linear projections with Xavier uniform
        self.q_proj.weight = init_fn(self.q_proj.weight)
        self.k_proj.weight = init_fn(self.k_proj.weight)
        self.v_proj.weight = init_fn(self.v_proj.weight)
        self.out_proj.weight = init_fn(self.out_proj.weight)
        
        # Initialize biases to zero if they exist
        if hasattr(self.q_proj, 'bias') and self.q_proj.bias is not None:
            self.q_proj.bias = mx.zeros_like(self.q_proj.bias)
        if hasattr(self.k_proj, 'bias') and self.k_proj.bias is not None:
            self.k_proj.bias = mx.zeros_like(self.k_proj.bias)
        if hasattr(self.v_proj, 'bias') and self.v_proj.bias is not None:
            self.v_proj.bias = mx.zeros_like(self.v_proj.bias)
        if hasattr(self.out_proj, 'bias') and self.out_proj.bias is not None:
            self.out_proj.bias = mx.zeros_like(self.out_proj.bias)
        
        # Layer norm weights are typically initialized to 1, bias to 0
        self.norm1.weight = mx.ones_like(self.norm1.weight)
        self.norm1.bias = mx.zeros_like(self.norm1.bias)
        self.norm2.weight = mx.ones_like(self.norm2.weight)
        self.norm2.bias = mx.zeros_like(self.norm2.bias)

    def __call__(self, query, key_value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]
        
        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        # Reshape and transpose for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # Attention computation
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            mask = mx.expand_dims(mask, axis=1)
            scores = mx.where(mask, scores, mx.full_like(scores, -float('inf')))
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = mx.matmul(attn_weights, V)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        return self.out_proj(attn_output)


# Example usage
if __name__ == "__main__":
    # Initialize cross-attention block
    d_model = 512
    n_heads = 8
    cross_attn = CrossAttentionBlock(d_model, n_heads)
    
    # Example tensors
    batch_size = 2
    seq_len_q = 10  # Query sequence length
    seq_len_kv = 20  # Key-Value sequence length
    
    query = mx.random.normal((batch_size, seq_len_q, d_model))
    key_value = mx.random.normal((batch_size, seq_len_kv, d_model))
    
    # Optional attention mask (1 for attend, 0 for ignore)
    mask = mx.ones((batch_size, seq_len_q, seq_len_kv))
    
    # Forward pass
    output = cross_attn(query, key_value, mask)
    print(f"Output shape: {output.shape}")  # Should be [batch_size, seq_len_q, d_model]
