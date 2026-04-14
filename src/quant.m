function [base_q,enh_q,enh_cnt,clip_cnt] = quant(base,enhance,Qb,Qe)
        base_q = round(base / Qb);% quant
        base_q = int16(base_q);
        enh_q  = round(enhance  / Qe);
         % enh -> int8 with clipping + stats
        enh_cnt = numel(enh_q);
        clip_cnt = sum(enh_q < -128 | enh_q > 127);
        enh_q = int8(max(min(enh_q,127),-128));
end