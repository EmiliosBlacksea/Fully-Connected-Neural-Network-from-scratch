import numpy as np


class FC_layer():
    def __init__(self, Input, Self, batch = 1, rng = None, LR = 10**(-4), activation = "relu", W = None, b = None):
        self.LR = LR
        self.batch = batch
        self.activation = activation
        self.step = 0
        self.b1 = 0.9
        self.b2 = 0.999
        if isinstance(rng, np.random.Generator):
            rng_gen = rng
        else:
            rng_gen = np.random.default_rng(rng)
        if W is None:                                                              #W = INPUT x SELF
            if activation == "relu":  # good for ReLU
                std = np.sqrt(2.0 / Input) 
                self.W = rng_gen.normal(0.0, std, size=(Input, Self)).astype(np.float32)
            if activation == "softmax" or activation == "sigmoid":  # good for sigmoid/tanh if 
                limit = np.sqrt(6.0 / max(1, (Input + Self)))
                self.W = rng_gen.uniform(-limit, limit, size=(Input, Self)).astype(np.float32)
            if activation == "nothing":
                self.W = rng_gen.normal(0.0, 0.01, size=(Input, Self)).astype(np.float32)
        if W is not None:
            self.W = W
        if b is None:
            self.b = np.zeros((1, Self), np.float32)
        if b is not None:
            self.b = b
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, IN):                                                          
        self.IN = IN
        self.z = IN @ self.W 
        self.z += self.b                                               
        self.a = self.func(self.z)                                                  
        return self.a
    
    def backward(self, C_y):   
        batch_size = C_y.shape[0]
        if self.activation == "softmax":#dL/dz = y * (dL/dy - sum[i](dLdy(i)*y(i))
            # inner = np.sum(C_y * self.a, axis=-1, keepdims=True)
            # C_z = self.a * (C_y - inner)  
            C_z = C_y
        else:                                             
            C_z = C_y * self.dfunc(self.z)
        dW = self.IN.T @ C_z
        db = np.sum(C_z, axis=0, keepdims=True)
        dW /= batch_size
        db /= batch_size
        
        C_a = C_z @ self.W.T  
        #update
        self.step += 1
        b1t = 1.0 - self.b1**self.step
        b2t = 1.0 - self.b2**self.step
        
        self.mW = self.b1 * self.mW + (1 - self.b1) * dW
        self.mb = self.b1 * self.mb + (1 - self.b1) * db
        self.vW = self.b2 * self.vW + (1 - self.b2) * (dW * dW)
        self.vb = self.b2 * self.vb + (1 - self.b2) * (db * db)
        
        mWhat = self.mW / b1t
        vWhat = self.vW / b2t
        mbhat = self.mb / b1t
        vbhat = self.vb / b2t
        
        self.W -= self.LR * (mWhat / (np.sqrt(vWhat) + 10**(-8)))
        self.b -= self.LR * (mbhat / (np.sqrt(vbhat) + 10**(-8))) 
        return C_a  

    def func(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.activation == "softmax":                    # keep dtype; or use float64 if you like
    # subtract row-wise max for numerical stability
            m = np.max(x, axis=-1, keepdims=True)
            z = x - m
            ez = np.exp(z)
            return ez / np.sum(ez, axis=-1, keepdims=True)
        if self.activation == "relu":
            return np.maximum(x, 0.0).astype(np.float32)
        if self.activation == "sigmoid":
            x_clip = np.clip(x, -60.0, 60.0)
            return (1.0 / (1.0 + np.exp(-x_clip))).astype(np.float32)
        if self.activation == "nothing":
            return x
    
    def dfunc(self, x : np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        
        if self.activation == "relu":
            return (x > 0).astype(np.float32)
        if self.activation == "sigmoid":
            x_clip = np.clip(x, -60.0, 60.0)
            s = 1.0 / (1.0 + np.exp(-x_clip))
            return (s * (1.0 - s)).astype(np.float32)
        if self.activation == "nothing":
            if x.ndim == 0:
                return np.ones((1, 1), dtype=np.float32)
            if x.ndim == 1:
                return np.ones((1, x.shape[0]), dtype=np.float32)
        return np.ones_like(x, dtype=np.float32)

    def get_W_b(self):
        return self.W, self.b



