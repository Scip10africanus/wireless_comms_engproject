function [base_hat,enh_hat] = dequant(base_q,enh_q,Qb,Qe)
    base_hat = double(base_q) * Qb;% dequant
    enh_hat  = double(enh_q)  * Qe;
end