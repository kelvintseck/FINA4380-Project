import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Span, HoverTool, CustomJS
from bokeh.models.widgets import Div
from bokeh.embed import file_html
from bokeh.resources import CDN
from math import pi
from bokeh.palettes import Category20

def create_portfolio_dashboard(portfolio, metrics_df, output_html: str = "portfolio_evaluation.html"):
    """
    Create an interactive HTML dashboard for portfolio evaluation using Bokeh.

    Args:
        portfolio: Portfolio object with historical data (value, returns, weights, etc.).
        metrics_df: DataFrame with portfolio metrics.
        output_html: Path to save the HTML file.
    """
    # Prepare data
    dates = portfolio.value.index
    weights_df = portfolio.weights
    portfolio_value = portfolio.value
    portfolio_cash = portfolio.cash

    # Convert dates to milliseconds for JavaScript compatibility
    x_dates = dates.to_pydatetime()
    x_dates_ms = [d.timestamp() * 1000 for d in x_dates]  # Convert to milliseconds

    # Create ColumnDataSource for line plots
    source = ColumnDataSource({
        'date': x_dates_ms,
        'Cumulative Return': metrics_df["Cumulative Return"],
        'Annualized Return': metrics_df["Annualized Return"],
        'Annualized Volatility': metrics_df["Annualized Volatility"],
        'VaR': metrics_df["VaR"],
        'Expected Shortfall': metrics_df["Expected Shortfall"],
        'Sharpe Ratio': metrics_df["Sharpe Ratio"],
        'Sortino Ratio': metrics_df["Sortino Ratio"],
        'Calmar Ratio': metrics_df["Calmar Ratio"],
        'Max Drawdown': metrics_df["Max Drawdown"],
        'Portfolio Value': portfolio_value,
        'Portfolio Cash': portfolio_cash,
        'Portfolio Asset Value': portfolio_value - portfolio_cash
    })

    # Create figures for line plots
    tools = "pan,box_zoom,wheel_zoom,reset,save"
    hover = HoverTool(tooltips=[("Date", "@date{%F}"), ("Value", "$y")], formatters={"@date": "datetime"})

    p1 = figure(title='Cumulative Return', x_axis_type="datetime", height=250, width=800, tools=tools, sizing_mode="scale_width")
    p1.line('date', 'Cumulative Return', source=source, legend_label="Cumulative Return", color="blue")
    p1.legend.click_policy = "hide"
    p1.legend.location = "top_left"
    
    p2 = figure(title='Annualized Return', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p2.line('date', 'Annualized Return', source=source, legend_label='Annualized Return', color="green")
    p2.legend.click_policy = "hide"
    p2.legend.location = "top_left"
    
    p3 = figure(title='Annualized Volatility', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p3.line('date', 'Annualized Volatility', source=source, legend_label='Annualized Volatility', color="red")
    p3.legend.click_policy = "hide"
    p3.legend.location = "top_left"
    
    p4 = figure(title='VaR', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p4.line('date', 'VaR', source=source, legend_label='VaR', color="orange")
    p4.legend.click_policy = "hide"
    p4.legend.location = "top_left"
    
    p5 = figure(title='Expected Shortfall', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p5.line('date', 'Expected Shortfall', source=source, legend_label='Expected Shortfall', color="purple")
    p5.legend.click_policy = "hide"
    p5.legend.location = "top_left"
    
    p6 = figure(title='Sharpe Ratio', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p6.line('date', 'Sharpe Ratio', source=source, legend_label='Sharpe Ratio', color="blue")
    p6.legend.click_policy = "hide"
    p6.legend.location = "top_left"
    
    p7 = figure(title='Sortino Ratio', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p7.line('date', 'Sortino Ratio', source=source, legend_label='Sortino Ratio', color="green")
    p7.legend.click_policy = "hide"
    p7.legend.location = "top_left"
    
    p8 = figure(title='Calmar Ratio', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p8.line('date', 'Calmar Ratio', source=source, legend_label='Calmar Ratio', color="cyan")
    p8.legend.click_policy = "hide"
    p8.legend.location = "top_left"

    p9 = figure(title='Max Drawdown', x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p9.line('date', 'Max Drawdown', source=source, legend_label='Max Drawdown', color="red")
    p9.legend.click_policy = "hide"
    p9.legend.location = "top_left"
    
    p10 = figure(title="Portfolio Value", x_axis_type="datetime", height=250, width=800, tools=tools, x_range=p1.x_range, sizing_mode="scale_width")
    p10.line('date', 'Portfolio Value', source=source, legend_label="Portfolio Value", color="black")
    p10.line('date', 'Portfolio Asset Value', source=source, legend_label="Portfolio Asset Value", color="orange")
    p10.legend.click_policy = "hide"
    p10.legend.location = "top_left"

    # Add hover tool to all plots
    for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]:
        p.add_tools(hover)

    # Initial vertical line (Span)
    initial_date_ms = x_dates_ms[0]
    vline = Span(location=initial_date_ms, dimension='height', line_color='gray', line_dash='dashed', line_width=1)
    for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]:
        p.add_layout(vline.clone())

    # Pie chart for weights
    initial_weights = weights_df.loc[dates[0]].fillna(0)
    n_assets = len(initial_weights)
    angles = initial_weights / initial_weights.sum() * 2 * pi
    end_angles = np.cumsum(angles)
    start_angles = np.roll(end_angles, 1)
    start_angles[0] = 0

    colors = Category20[20]
    colors = (colors * (n_assets // len(colors) + 1))[:n_assets]  # Repeat colors if needed

    pie_source = ColumnDataSource({
        'start_angle': start_angles,
        'end_angle': end_angles,
        'color': colors,
        'label': initial_weights.index,
        'value': initial_weights
    })

    p11 = figure(title=f"Weights on {dates[0].strftime('%Y-%m-%d')}", height=300, width=400, tools="hover",
                tooltips="@label: @value", x_range=(-1, 1), y_range=(-1, 1))
    p11.wedge(x=0, y=0, radius=0.8, start_angle='start_angle', end_angle='end_angle', color='color', source=pie_source)

    # Prepare weights data for JavaScript (convert to list of lists)
    weights_data = [weights_df.loc[date].fillna(0).tolist() for date in dates]
    dates_str = [date.strftime('%Y-%m-%d') for date in dates]

    # Slider with JavaScript callback
    slider = Slider(start=0, end=len(dates) - 1, value=0, step=1, title="Day Index")

    callback = CustomJS(args=dict(
        vline=vline,
        pie_source=pie_source,
        p11=p11,
        dates=x_dates_ms,
        dates_str=dates_str,
        weights=weights_data,
        labels=weights_df.columns.tolist(),
        colors=colors
    ), code="""
        const idx = cb_obj.value;
        const selected_date = dates[idx];
        vline.location = selected_date;  // Update vertical line

        // Update pie chart
        const weights_at_idx = weights[idx];
        const total = weights_at_idx.reduce((a, b) => a + b, 0);
        const angles = weights_at_idx.map(w => (w / total) * 2 * Math.PI);
        const end_angles = [];
        let sum = 0;
        for (let i = 0; i < angles.length; i++) {
            sum += angles[i];
            end_angles.push(sum);
        }
        const start_angles = [0].concat(end_angles.slice(0, -1));

        pie_source.data['start_angle'] = start_angles;
        pie_source.data['end_angle'] = end_angles;
        pie_source.data['value'] = weights_at_idx;
        pie_source.data['label'] = labels;
        pie_source.data['color'] = colors.slice(0, weights_at_idx.length);
        pie_source.change.emit();

        p5.title.text = `Weights on ${dates_str[idx]}`;
    """)

    slider.js_on_change('value', callback)

    # Layout
    layout = column(
        Div(text="<h2>Portfolio Metrics Dashboard</h2>"),
        gridplot([
            [p1, p2],
            [p3, p4],
            [p5, p6],
            [p7, p8],
            [p9, p10],
        ], sizing_mode="stretch_both"),
        p11,
        slider,
        sizing_mode="stretch_width"
    )


    # Save to HTML
    html = file_html(layout, CDN, "Portfolio Dashboard")
    with open(output_html, 'w') as f:
        f.write(html)



if __name__ == "__main__":
    pass