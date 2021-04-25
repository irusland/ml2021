class Linear():
    def __init__(self, n_in, n_out):#инициализацию трогать не надо
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in)) # веса - произвольные
        self.b = np.random.uniform(-stdv, stdv, size = n_out)      
        self.gradW = np.zeros_like(self.W) # градиенты изначально нули
        self.gradb = np.zeros_like(self.b) 
        
        
    def forward(self, input: np.ndarray): #input batch x n_in
        self.output = (input @ self.W.T) + self.b
        return self.output
    
    def backward(self, gradErrors: np.ndarray): #gradOutput batch x n_out
        self.gradInput = gradErrors @ self.W
        return self.gradInput
    
    def findGrads(self, input: np.ndarray, gradErrors: np.ndarray): #gradOutput dL/dy
        self.gradW = gradErrors.T @ input
        self.gradb = np.sum(gradErrors, axis=0)