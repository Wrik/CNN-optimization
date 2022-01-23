clc
clear
close all
%%
load('LLT_Data_test.mat', 'CL','CDi');
%%
[CLpred_test,CDipred_test] = calc_test_pred('./');
%%
RsqCD = calcRsquare(CLpred_test,CL);
RsqCL = calcRsquare(CDipred_test,CDi);
%%
figure(101)
hold on
grid on
scatter(CLpred_test,CL,200,'.')
x = linspace(min(CL), max(CL));
y = x;
plot(x, y, '-', 'linewidth', 2)
xlabel('$C_L$ target')
ylabel('$C_L$ predicted')
hleg1 = legend('Pred','Truth');
set(hleg1,'FontName','Arial','FontSize',12,'Interpreter','latex')
ax = gca;
ylim([-0.2 1.2])
xlim([-0.2 1.2])
xticks(-0.2:0.2:1.2)
yticks(-0.2:0.2:1.2)
ax.FontSize = 16;
hold off
%%
figure(102)
hold on
grid on
scatter(CDipred_test,CDi,200,'.')
x = linspace(min(CDi), max(CDi));
y = x;
plot(x, y, '-', 'linewidth', 2)
xlabel('$C_D$ target')
ylabel('$C_D$ predicted')
hleg1 = legend('Pred','Truth');
set(hleg1,'FontName','Arial','FontSize',12,'Interpreter','latex')
ax = gca;
ax.FontSize = 16;
ylim([0 0.065])
xlim([0 0.065])
xticks(0:0.01:0.06)
yticks(0:0.01:0.06)
hold off
%%
function [CLpred_test,CDpred_test]= calc_test_pred(path)
load 'Geometry_Set_test.mat'   geo_set
load(strcat(path,'wingCNN_CL.mat'), 'wingCNN_CL')
test_input = (geo_set-wingCNN_CL.avg)/sqrt(wingCNN_CL.variance);
network_CL = wingCNN_CL.network;
var_CL = wingCNN_CL.var_CL;
avg_CL = wingCNN_CL.avg_CL;
load(strcat(path,'wingCNN_CD.mat'), 'wingCNN_CD')
network_CD = wingCNN_CD.network;
var_CD = wingCNN_CD.var_CD;
avg_CD = wingCNN_CD.avg_CD;

test_outputCL = predict(network_CL, test_input);
test_outputCD = predict(network_CD, test_input);
CLpred_test = test_outputCL*sqrt(var_CL) + avg_CL;
CDpred_test = test_outputCD*sqrt(var_CD) + avg_CD;
end