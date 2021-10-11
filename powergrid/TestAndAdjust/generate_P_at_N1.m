function P_at_N5 = generate_P_at_N5()

global margin_upper margin_lower P_beg P_end m;

P_at_N5 = zeros(m,1);
slope = (abs(P_end)-abs(P_beg))/(margin_upper - margin_lower);
c = abs(P_beg) - slope*margin_lower;

for t = 1:m
    if t < margin_lower
        P_at_N5(t) = P_beg;
    elseif (margin_lower <= t) && (t < margin_upper)
        P_at_N5(t) = -(slope*t + c);
    else
        P_at_N5(t) = P_end;
    end
end
end