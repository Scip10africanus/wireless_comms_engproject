function [base,enhance] = Layer_Split(k,N,block)
    index = Zigzag_index(N);     
    vec = block(index);
    base = vec(1:k);
    enhance = vec(k+1:end);
end