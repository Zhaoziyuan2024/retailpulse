"""Build an offline business report from UCI-format transactions."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retailpulse.core import clean_transactions, cohort_retention, evaluate, features


def segment_customers(frame):
    """Transparent RFM rules; quantile thresholds are descriptive, not learned labels."""
    result = frame.copy()
    high_value = result.monetary >= result.monetary.quantile(.75)
    result["segment"] = np.select(
        [(result.recency <= 30) & (result.frequency >= 3) & high_value,
         (result.recency > 60) & high_value,
         (result.recency <= 30) & (result.frequency == 1)],
        ["Champions", "High-value dormant", "Recent one-time"], default="Other active")
    return result


def svg(fig):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return "<svg" + buffer.getvalue().split("<svg", 1)[1]


def build_report(df, audit, evaluation, cohort, segments, digest, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    monthly = df.groupby(df.InvoiceDate.dt.to_period("M")).revenue.sum()
    # Drop the final partial month from trend comparisons, retaining it in all-time KPIs.
    complete_monthly = monthly[monthly.index < df.InvoiceDate.max().to_period("M")]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(complete_monthly.index.astype(str), complete_monthly.values, "o-", color="#0d9488")
    ax.set(ylabel="Positive sales (GBP)", title="Sales in complete calendar months")
    ax.tick_params(axis="x", rotation=35)
    sales_svg = svg(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    heat = ax.imshow(cohort.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cohort.columns)), cohort.columns)
    ax.set_yticks(range(len(cohort.index)), cohort.index)
    ax.set(xlabel="Months since first observed purchase", title="Cohort retention (blank = not observed)")
    fig.colorbar(heat, ax=ax, label="Customer retention")
    cohort_svg = svg(fig)
    metrics = pd.DataFrame({"Logistic regression": evaluation["model_metrics"],
                            "Recency baseline": evaluation["baseline_metrics"]})
    summary = segments.groupby("segment").agg(customers=("monetary", "size"),
                                               sales_gbp=("monetary", "sum"))
    summary["sales_share"] = summary.sales_gbp / summary.sales_gbp.sum()
    summary.to_csv(out / "segments.csv")
    orders = df.InvoiceNo.nunique()
    model = evaluation["model_metrics"]
    baseline = evaluation["baseline_metrics"]
    decision = ("模型在本次测试的前 20% 客户精确率超过简单规则，可作为实验候选。"
                if model["precision_at_20"] > baseline["precision_at_20"] else
                "模型在本次测试的前 20% 客户精确率没有超过简单规则，应优先保留简单规则。")
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RetailPulse · Business Analytics</title>
<style>body{{margin:0;background:#f4f7fa;color:#172b3a;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:36px 24px}}header{{padding:36px;background:#102f44;color:white;border-radius:18px}}h1{{font-size:42px;margin:4px 0}}h2{{margin-top:0}}.muted{{color:#556a78}}section{{background:white;border:1px solid #dce5eb;border-radius:14px;padding:26px;margin:22px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:22px 0}}.card{{background:white;padding:22px;border-radius:12px}}.big{{font-size:27px;font-weight:700;color:#0f766e}}svg{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #e5edf2}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}</style>
<main><header><small>BUSINESS ANALYTICS / REPRODUCIBLE CASE STUDY</small><h1>RetailPulse</h1><p>从交易记录到客户运营决策 · 真实数据 / 时间验证 / 明确边界</p></header>
<div class="cards"><div class="card">有效客户<div class="big">{df.CustomerID.nunique():,}</div></div>
<div class="card">有效订单<div class="big">{orders:,}</div></div><div class="card">正向销售额（GBP）<div class="big">£{df.revenue.sum():,.0f}</div></div>
<div class="card">订单均额（GBP）<div class="big">£{df.revenue.sum()/orders:,.2f}</div></div></div>
<section><h2>01 / 业务结论</h2><p>{decision}</p><p>测试集前 20% 客户的实际复购率为 <b>{model['precision_at_20']:.1%}</b>，相对随机选择的 lift 为 <b>{model['lift_at_20']:.2f}×</b>。这是历史预测能力，并非营销带来的增量效果；11 月节日效应也可能影响结论。</p><p>下一步：对候选客户随机分配触达组与不触达组，预先设定 30 天增量毛利为主指标、退订率为护栏，并按预算及最小可检测效应计算样本量。</p></section>
<section><h2>02 / 销售趋势与留存</h2>{sales_svg}{cohort_svg}<p class="muted">留存定义：某月首次被观察到的客户，在第 n 月再次购买的人数 / 该 cohort 客户数。不是连续留存，也不能认定首次观察就是首次购买。完整月只到 2011-11，12 月被排除。</p></section>
<section><h2>03 / RFM 客户运营</h2><p>快照 2011-11-01；仅分析此前 90 天有购买的客户。高价值定义为该快照金额的上四分位。</p><div class="scroll">{summary.to_html(float_format=lambda x: f'{x:,.3f}', border=0)}</div><p>Champions：近期多次购买且高价值，优先服务体验；High-value dormant：60 天未购买但高价值，适合设计召回实验；Recent one-time：近期首次窗口内购买，适合测试第二次购买引导。</p><p class="muted">“one-time”仅表示观察窗口内一单，不等于全生命周期新客；超过 90 天不活跃客户不在本模型覆盖范围。</p></section>
<section><h2>04 / 复购预测与基线</h2><div class="scroll">{metrics.to_html(float_format=lambda x: f'{x:.4f}', border=0)}</div><p>训练、验证、测试按时间划分，分别预测 7 月、9 月和 11 月的未来 30 天购买。验证集选择正则化强度，测试集不参与选择。recency baseline 按距最近购买时间排序。</p><div class="scroll">{pd.DataFrame(evaluation['split_sizes']).T.to_html(border=0)}</div><p class="muted">同一客户可跨时间快照出现，模拟对已有客户反复评分；CustomerID 不作为特征。单个留出月份不代表跨行业泛化，无置信区间，微小差异不作显著性判断。</p></section>
<section><h2>05 / 数据审计与复现</h2><p>已排除取消、非正数量/价格、缺失客户及重复行。这里的金额是符合条件交易的正向销售额，未扣退货，不能当净收入或利润。</p><div class="scroll">{pd.Series(audit, name='value').to_frame().to_html(border=0)}</div><p>输入文件 SHA256：<code>{digest}</code></p><p>来源：<a href="https://archive.ics.uci.edu/dataset/352/online-retail">UCI Online Retail</a>，Chen (2015)，DOI:10.24432/C5BW33，CC BY 4.0。历史英国零售数据，包含批发客户，不代表当前市场。</p></section></main></html>'''
    (out / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        raw = pd.read_csv(args.input) if args.input.suffix.lower() == ".csv" else pd.read_excel(args.input)
        df, audit = clean_transactions(raw)
        evaluation = evaluate(df)
        cohort = cohort_retention(df)
        segments = segment_customers(features(df, "2011-11-01"))
        digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
        args.out.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(args.out / "cohort_retention.csv")
        payload = {"input_sha256": digest, "audit": audit, **evaluation}
        (args.out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        build_report(df, audit, evaluation, cohort, segments, digest, args.out)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"retailpulse: {exc}\n")
    print(f"Report: {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
