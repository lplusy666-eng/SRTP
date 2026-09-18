% 核心年份数据 (X轴需要使用数值或分类数组，此处为方便绘图使用数值)
years = ;

% 全社会用电量同比增长率 (%)
electricity_growth = [6.2, 2.8, 9.5, 15.6, 5.6, 7.2, 7.5, 5.0, 4.5, 3.1, 3.6, 6.7, 6.8];

% 实际GDP同比增长率 (%)
gdp_growth = [3.8, 7.8, 8.4, 10.0, 9.6, 9.1, 7.8, 6.8, 6.0, 2.2, 3.0, 5.2, 5.0];

% 创建图形窗口
figure('Position', );

% 绘制全社会用电量增速折线（蓝色，带圆形标记）
plot(years, electricity_growth, '-o', 'Color', [0 0.4470 0.7410],...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0 0.4470 0.7410]);
hold on;

% 绘制实际GDP增速折线（绿色，带方形标记）
plot(years, gdp_growth, '-s', 'Color', [0.4660 0.6740 0.1880],...
    'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', [0.4660 0.6740 0.1880]);

% 添加图表标题和坐标轴标签
title('中国实际GDP增速与全社会用电量增速历史同步趋势与背离分析', 'FontSize', 14);
xlabel('年份', 'FontSize', 12);
ylabel('年度同比增长率 (%)', 'FontSize', 12);

% 自定义 X 轴的刻度和显示标签，以匹配非连续的年份数据
xticks(years);
xticklabels({'1990', '1998', '2000', '2003', '2008', '2009', '2013', '2016', '2019', '2020', '2022', '2023', '2024'});

% 添加图例
legend('全社会用电量增速 (%)', '实际GDP增速 (%)', 'Location', 'northeast', 'FontSize', 11);

% 开启网格线
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.6);

hold off;