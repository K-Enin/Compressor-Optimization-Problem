function Q_at_N5 = generate_Q_at_N5()

global margin_upper margin_lower Q_beg Q_end m;

Q_at_N5 = zeros(m,1);
slope = (abs(Q_end)-abs(Q_beg))/(margin_upper - margin_lower);
c = abs(Q_beg) - slope*margin_lower;

for t = 1:m
    if t < margin_lower
        Q_at_N5(t) = Q_beg;
    elseif (margin_lower <= t) && (t < margin_upper)
        Q_at_N5(t) = -(slope*t + c);
    else
        Q_at_N5(t) = Q_end;
    end
end
end