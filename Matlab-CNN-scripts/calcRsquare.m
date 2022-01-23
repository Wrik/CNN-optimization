function Rsquare = calcRsquare(vec1,vec2)
sse = sum((vec1-vec2).^2);
Rsquare = 1 - sse/(length(vec1)*var(vec2));
end