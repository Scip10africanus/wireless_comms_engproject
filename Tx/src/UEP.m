function [waveTx, cfg, meta] = UEP(pkt_kind, pkt_bytes_base, pkt_bytes_enh, pkt_bytes_combined, mcs_base, mcs_enh)
% pkt_kind: "combined" | "base" | "enh"
    pkt_kind = string(pkt_kind);
    switch pkt_kind
        case "combined" % combined packet case 
            assert(~isempty(pkt_bytes_combined), "combined selected but pkt_bytes_combined is empty.");
            [waveTx, cfg] = modem(pkt_bytes_combined, mcs_base);  % conservative by design
            meta.kind = "combined";
            meta.mcs_used = double(mcs_base);
            meta.psdu_len_bytes = numel(pkt_bytes_combined);

        case "base" %  base only in split layer mode case 
            assert(~isempty(pkt_bytes_base), "base selected but pkt_bytes_base is empty.");
            [waveTx, cfg] = modem(pkt_bytes_base, mcs_base);
            meta.kind = "base";
            meta.mcs_used = double(mcs_base);
            meta.psdu_len_bytes = numel(pkt_bytes_base);

        case "enh" % enhance only in split layer mode case
            assert(~isempty(pkt_bytes_enh), "enh selected but pkt_bytes_enh is empty.");
            [waveTx, cfg] = modem(pkt_bytes_enh, mcs_enh);
            meta.kind = "enh";
            meta.mcs_used = double(mcs_enh);
            meta.psdu_len_bytes = numel(pkt_bytes_enh);

        otherwise
            error("Unknown pkt_kind: %s", pkt_kind);
    end
end