function [indexes] = Zigzag_index(n)
order = zeros(n, n); % build a n by n block to browse index
current_number = 1;
for s = 1:(2*n-1) 
        if mod(s,2)==0
            i_start = max(1, s-n+1);
            i_end   = min(n, s);
            for i = i_start:i_end
                j = s - i + 1;
                order(i,j) = current_number; 
                current_number = current_number + 1;
            end
        else
            i_start = min(n, s);
            i_end   = max(1, s-n+1);
            for i = i_start:-1:i_end
                j = s - i + 1;
                order(i,j) = current_number; 
                current_number = current_number + 1;
            end
        end
end

indexes = zeros(n*n,1); % [n*n, 1] for output 
for p = 1:n*n
    [row,col] = find(order==p,1); % find index in [n,n]
    indexes(p) = sub2ind([n,n], row, col); % row first transform 
end
end