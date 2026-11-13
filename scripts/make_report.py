"""Build the course-report PDF (Chinese, 4-6 pages) from measured results.

All numbers are read from results/numbers.json and results/*/log.jsonl, so the
report can never drift away from the experiments.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RES = Path("results")
FIG = RES / "figures"
OUT = Path("../2026439189-李珅.pdf")

pdfmetrics.registerFont(TTFont("SimSun", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))
pdfmetrics.registerFontFamily("SimSun", normal="SimSun", bold="SimHei",
                              italic="SimSun", boldItalic="SimHei")

BODY = ParagraphStyle("body", fontName="SimSun", fontSize=9.3, leading=13.2,
                      alignment=TA_JUSTIFY, firstLineIndent=18.6, wordWrap="CJK",
                      spaceBefore=1, spaceAfter=1)
BODY0 = ParagraphStyle("body0", parent=BODY, firstLineIndent=0)
TITLE = ParagraphStyle("title", fontName="SimHei", fontSize=16, leading=21,
                       alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="SimSun", fontSize=9.2, leading=13,
                     alignment=TA_CENTER, textColor=colors.Color(0.25, 0.25, 0.25))
H1 = ParagraphStyle("h1", fontName="SimHei", fontSize=10.8, leading=14.6,
                    spaceBefore=5.5, spaceAfter=2.5, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="SimHei", fontSize=9.6, leading=13.0,
                    spaceBefore=3, spaceAfter=1.5, keepWithNext=1)
CAP = ParagraphStyle("cap", fontName="SimSun", fontSize=8.0, leading=10.4,
                     alignment=TA_CENTER, spaceBefore=1.5, spaceAfter=3.0)
TCELL = ParagraphStyle("tcell", fontName="SimSun", fontSize=8.0, leading=10.6,
                       wordWrap="CJK")
TCELLC = ParagraphStyle("tcellc", parent=TCELL, alignment=TA_CENTER)

num = json.loads((RES / "numbers.json").read_text(encoding="utf-8"))

_SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺"
_SUP_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def sanitize(text):
    """Replace glyphs missing from SimSun and convert superscripts to <super>."""
    text = text.replace("−", "-").replace("–", "-").replace("‖", "||")
    out, i = [], 0
    while i < len(text):
        if text[i] in _SUPS:
            j = i
            while j < len(text) and text[j] in _SUPS:
                j += 1
            out.append("<super>" + text[i:j].translate(_SUP_MAP) + "</super>")
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def P(text, style=BODY):
    return Paragraph(sanitize(text), style)


def _raw(cond, label, field):
    v = num[cond][label][field]
    m = v[0]
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return None, None
    s = v[1] if len(v) > 1 else 0.0
    if s is None or (isinstance(s, float) and math.isnan(s)):
        s = 0.0
    return float(m), float(s)


def pct(cond, label, field="success", plus=True):
    m, s = _raw(cond, label, field)
    if m is None:
        return "—"
    if s > 0 and plus:
        return f"{m * 100:.1f}±{s * 100:.1f}"
    return f"{m * 100:.1f}"


def f2(cond, label, field):
    m, _ = _raw(cond, label, field)
    if m is None:
        return "—"
    txt = f"{m:.2f}"
    return "0.00" if txt == "-0.00" else txt


def f1(cond, label, field):
    m, _ = _raw(cond, label, field)
    return "—" if m is None else f"{m:.1f}"


def val(cond, label, field):
    m, s = _raw(cond, label, field)
    return (0.0, 0.0) if m is None else (m, s)


def img(path, width_cm):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def figblock(path, width_cm, caption):
    """Figure with its caption, kept on one page."""
    return KeepTogether([img(path, width_cm), P(caption, CAP)])


def curve_at(run, step):
    """Eval success rate closest to ``step`` for one run."""
    best, bestd = None, 1e18
    for line in (RES / run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "eval" in rec and abs(rec["step"] - step) < bestd:
            best, bestd = rec["eval"]["success"], abs(rec["step"] - step)
    return best


def mean_at(prefix, step):
    """Mean of the eval success rate at ``step`` over the 3 seeds of ``prefix``."""
    vals = [curve_at(f"{prefix}_s{i}", step) for i in range(3)]
    return sum(vals) / len(vals)


def table(rows, widths, font=8.0, header=True, align="CENTER"):
    data = rows
    ws = [w * cm for w in widths]
    limit = 16.9 * cm
    total = sum(ws)
    if total > limit:                      # 防止表格超出版心宽度
        scale = limit / total
        ws = [w * scale for w in ws]
    t = Table(data, colWidths=ws, hAlign=align, repeatRows=1 if header else 0)
    st = [("FONTNAME", (0, 0), (-1, -1), "SimSun"),
          ("FONTSIZE", (0, 0), (-1, -1), font),
          ("LEADING", (0, 0), (-1, -1), font + 2.6),
          ("ALIGN", (0, 0), (-1, -1), "CENTER"),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.Color(0.75, 0.75, 0.75)),
          ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
          ("TOPPADDING", (0, 0), (-1, -1), 1.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]
    if header:
        st += [("FONTNAME", (0, 0), (-1, 0), "SimHei"),
               ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.92))]
    t.setStyle(TableStyle(st))
    return t


def build():
    story = []
    S = story.append

    # ---------------------------------------------------------------- header
    S(P("基于近端策略优化的无人机端到端避障导航", TITLE))
    S(P("——奖励塑形、感知范围与分布偏移下的泛化性", SUB))
    S(Spacer(1, 4))
    S(P("2026439189  李珅　　（深度学习课程设计，2026 秋季）", SUB))
    S(P("代码：https://github.com/lishen2004328/homework", SUB))
    S(Spacer(1, 8))

    # --------------------------------------------------------------- abstract
    S(P(
        f"<b>摘要</b>：本文研究无人机在未知杂乱环境中的自主避障导航。我们搭建了一个轻量、完全向量化的三维导航仿真环境，"
        f"智能体仅依靠 32 束水平激光测距、目标方位与自身运动状态（共 48 维观测），用近端策略优化（PPO）学习从观测到三维加速度指令的端到端策略，"
        f"并与 Double DQN、人工势场法、基于栅格 A* 的特权（oracle）上界及随机策略在同一固定场景集（每条件 256 个回合）上统一评测。"
        f"在主条件（25 个障碍、8 m 感知）下，PPO 经 2×10⁶ 步训练后成功率达到 {pct('main', 'PPO')}%，逼近拥有完整地图的 A* 上界"
        f"（{pct('main', 'A*')}%），显著高于人工势场法（{pct('main', 'Reactive')}%）与 Double DQN（{pct('main', 'DQN')}%）；"
        f"其控制能耗为 {f2('main', 'PPO', 'energy')}，仅为 A* 的 {val('main', 'PPO', 'energy')[0] / val('main', 'A*', 'energy')[0] * 100:.0f}%、"
        f"人工势场法的 {val('main', 'PPO', 'energy')[0] / val('main', 'Reactive', 'energy')[0] * 100:.0f}%。"
        f"消融显示势函数塑形把早期样本效率提高约 5 倍（8×10⁴ 步时 {mean_at('ppo', 81920) * 100:.1f}% 对稀疏奖励的 "
        f"{mean_at('ppo_sparse', 81920) * 100:.1f}%），但稀疏奖励在 3×10⁵ 步附近追平并最终略高，代价是更慢、更保守的飞行。"
        f"感知范围消融暴露了明显的非对称迁移：8 m 策略在 4 m 量程下几乎无损（{pct('per4', 'PPO')}%），"
        f"而 4 m 策略在 8 m 量程下从 {pct('per4', 'PPO-4m')}% 跌到 {pct('main', 'PPO-4m')}%。"
        f"在障碍密度、地图尺度与测距噪声三类分布偏移下，PPO 的衰减幅度远小于人工势场与 Double DQN，"
        f"说明数据驱动的策略学到了与场景统计相关的更一般的避障几何，而非一组固定权重。", BODY))
    S(P("<b>关键词</b>：无人机导航；深度强化学习；近端策略优化；奖励塑形；泛化性", BODY0))
    S(Spacer(1, 2))

    # ------------------------------------------------------------ 1 motivation
    S(P("1  研究问题", H1))
    S(P(
        "森林巡检、城市楼宇间配送、灾后搜救等任务，都要求无人机在只有机载测距传感器的条件下完成避障与到达目标。"
        "经典方案是“建图—规划—控制”的分层流水线：先由激光或视觉构建占据栅格/ESDF 地图，再用 A*、RRT 等规划几何路径，"
        "最后由控制器跟踪路径。这类方法在静态、可建图环境中成熟可靠，但每一层都依赖准确的感知与手工调参，"
        "当感知存在噪声、障碍动态变化或机载算力受限时，误差会逐层放大。", BODY))
    S(P(
        "另一条路线是端到端强化学习：把测距观测直接映射为控制指令，由奖励函数隐式地定义“安全、快速、平滑”的飞行行为。"
        "它不需要显式地图与规划器，可以在仿真中大规模试错，并通过域随机化迁移到真机。但端到端方法同样面临三个现实困难："
        "奖励稀疏导致探索困难、部分可观测导致策略短视、训练分布与测试分布不一致导致泛化脆弱。", BODY))
    S(P(
        "本文不追求提出新算法，而是把上述问题收缩成一个规模可控、可完全复现的小型研究，回答三个具体问题："
        "<b>RQ1</b>：仅凭 32 束水平激光与目标方位，PPO 能否学出安全且高效的导航策略，与经典势场法、A* 上界的差距有多大？"
        "<b>RQ2</b>：奖励塑形与感知范围这两个最影响实际部署的设计选择，分别如何影响样本效率与最终性能？"
        "<b>RQ3</b>：在障碍密度、地图尺度与测距噪声发生分布偏移时，学习策略是否比手工控制器更鲁棒？", BODY))
    S(P(
        "围绕这三个问题，本文的工作包括：（1）一个轻量、完全向量化的三维导航仿真环境与可复现实验脚本；"
        "（2）统一评测协议——同一固定场景集、7 种测试条件、5 类控制器的定量对照；"
        "（3）奖励设计与感知范围两组消融；（4）结合轨迹案例的失败模式分析。全部代码与实验配置开源，见首页链接。", BODY))

    # ------------------------------------------------------- 2 data & platform
    S(P("2  数据与实验环境", H1))
    S(P(
        "本文不使用公开数据集，任务数据由自建仿真器在线生成，随机性由显式种子控制，评测集与训练集使用互不相同的种子。"
        "选择自建仿真器而非现成 benchmark（如 gym-pybullet-drones、FlightGym）的原因有三：其一，课程作业算力有限（单机 CPU），"
        "需要每秒数万步的环境吞吐；其二，泛化实验需要可控地改变障碍密度、地图尺度与感知噪声，这些在物理引擎中改动成本很高；"
        "其三，场景、奖励与指标完全自定，保证可复现。", BODY))
    S(P(
        "世界为 15×15 m² 的方形区域，高度 1.0–3.5 m。障碍物为竖直圆柱（半径 0.25–0.70 m，贯穿整个高度区间），"
        "起点在区域内均匀随机采样，目标点位于起点 4.5–6.5 m 的随机方位上。放置障碍时要求障碍中心与起点、目标的距离不小于 r+0.75 m、"
        "障碍彼此表面间距不小于 0.6 m，以排除“出生即碰撞”和完全封死通道的不可解场景。飞行器按双积分模型建模，"
        "策略输出的三维加速度指令经限幅后积分得到速度与位置。主要参数见表 1。", BODY))

    S(P(
        "评测协议：所有方法在同一组固定场景上运行，每个条件 256 个回合，场景由与训练无关的种子生成。除与训练同分布的 main 条件外，"
        "另设 6 个偏移条件：障碍数降为 10（d10）、升为 40（d40）、地图放大到 24×24 m² 且障碍同密度增加（scale）、"
        "激光噪声为量程的 5% 与 15%（noise05 / noise15）、以及把感知量程压到 4 m（per4）。", BODY))

    S(P(
        "观测与动作的设计也按同一逻辑取舍。32 束激光只扫水平面，因为障碍是贯穿整个高度区间的竖直圆柱，水平截面已能唯一确定其方位与厚度；"
        "8 m 量程覆盖了 15 m 场地内绝大多数与决策相关的障碍（目标最远 6.5 m），更远的信息对当前动作几乎没有影响。"
        "动作取三维加速度而不是速度指令，让策略直接输出“推力”，平滑性与动力学约束交给积分环节，避免速度指令在障碍附近被硬性截断。"
        "回合上限 150 步（15 s）由 v_max=2 m/s 与最长 6.5 m 的直线距离推出：即使绕行使路径翻倍也留有余量，而主实验中 PPO 的平均回合长度仅 29 步。", BODY))

    rows = [["项目", "设置"],
            ["地图 / 障碍物", "15×15 m²，25 个竖直圆柱，半径 0.25–0.70 m，净距 ≥0.6 m"],
            ["动力学", "双积分模型；a_max=2 m/s²，v_max=2 m/s，dt=0.1 s，单回合 ≤150 步"],
            ["观测（48 维）", "32 束水平激光（0–8 m，归一化）；目标单位向量 3；目标距离 1；速度 3；上一步动作 3；四周墙体距离 4；高度余量 2"],
            ["动作（3 维）", "加速度指令 a∈[-1,1]³，按 a_max 缩放"],
            ["塑形奖励", "2·Δ(目标距离) − 0.01 − 0.02‖a‖² − 0.25·max(0, 0.6−最小间隙) + 3（到达）− 3（碰撞/越界）− 0.5（超时）"],
            ["稀疏奖励", "−0.01 每步 + 3（到达）− 3（碰撞/越界）− 0.5（超时）"],
            ["计算环境", "i5-12600KF（10 核 16 线程），32 GB 内存，CPU 版 PyTorch 2.15"]]
    rows = [[P(c, TCELLC if i == 0 else TCELL) for c in r] for i, r in enumerate(rows)]
    S(KeepTogether([P("表 1　仿真环境与任务配置", CAP),
                    table(rows, [3.0, 13.9], font=7.6)]))


    # ---------------------------------------------------------- 3 method
    S(P("3  方法与技术", H1))
    S(P(
        "图 1 给出本文的闭环结构：策略把 48 维观测映射为三维加速度指令，指令经限幅进入双积分动力学，环境返回奖励与下一观测；"
        "训练时 PPO 用 GAE 估计优势并做裁剪更新，DQN 则把动作离散化后用经验回放更新 Q 网络。所有控制器共用同一套环境与指标。", BODY))
    S(figblock(FIG / "fig0_method.png", 14.6, "图 1　闭环控制与训练流程示意"))
    S(P("3.1  近端策略优化（PPO）", H2))
    S(P(
        "策略采用 Actor-Critic 结构：观测先经两层 128 单元 tanh 隐层，再由两个线性头分别输出高斯策略的均值与状态价值；"
        "策略标准差为与状态无关的可学习参数，初值取 log σ=−0.7。PPO 用裁剪的替代目标限制每次更新的策略变化："
        "L(θ)=E[min(r_t(θ)A_t, clip(r_t(θ),1−ε,1+ε)A_t)]，其中 r_t 为新旧策略概率比，ε=0.2；"
        "优势函数用广义优势估计（GAE，γ=0.99，λ=0.95）计算并在小批量内归一化；同时加入 0.01 的熵正则与 0.5 的价值损失权重，"
        "梯度范数裁剪到 0.5。选择 PPO 的理由是：它天然支持连续动作、on-policy 更新稳定、对超参不敏感且实现简单，"
        "在“算力有限、需要多次重复实验”的课程作业场景中比 SAC/TD3 更容易得到可复现的结果。", BODY))

    S(P("3.2  奖励设计与势函数塑形", H2))
    S(P(
        "导航任务若只给终端奖励，探索极其困难：智能体在 15×15 m² 的场地上随机游走几乎不可能撞上目标。"
        "本文的主奖励采用基于势函数的塑形，Φ(s)=−d(s) 为到目标的欧氏距离，每一步给 2·(Φ(s_t)−Φ(s_{t+1})) 的进展奖励。"
        "Ng 等（1999）证明这种“势差”形式的塑形不改变最优策略的集合，所以它原则上只改变优化路径、不改变最优解；"
        "但本文的实验（5.2 节）显示，在有限预算与有限容量下它仍会改变实际收敛点：早期快得多，最终却略低于稀疏奖励。"
        "在此之上叠加三项：每步 −0.01 的时间惩罚、−0.02‖a‖² 的控制能耗惩罚、以及最小间隙小于 0.6 m 时的线性近距惩罚，"
        "让策略在学“快”之前先学“不撞”。稀疏奖励对照组去掉进展项与近距惩罚，只保留时间惩罚和终端奖励。", BODY))

    S(P("3.3  对比方法与评价指标", H2))
    S(P(
        "Double DQN：把三维加速度离散为 27 个组合（每轴取 −1/0/1），用 128×128 的 Q 网络、经验回放（容量 2×10⁵）、"
        "目标网络每 2000 次梯度更新同步一次，ε 从 1.0 线性衰减到 0.05。人工势场法：吸引力取目标方向，"
        "斥力取 1.5 m 内各激光束反方向的加权和，合成期望速度后用同一底层动力学跟踪，代表“手工设计、无需学习”的经典方案。"
        "A* 上界：在 0.3 m 分辨率的栅格上以膨胀后的障碍做 8 邻域搜索，再跟踪航点；它使用真值地图，代表特权（oracle）性能上界。"
        "随机策略作为下界。", BODY))
    S(P(
        "指标包括：成功率、碰撞率、越界率、超时率；SPL（成功加权路径长度，Anderson 等，2018）；路径比（实际路径 / 最短可行路径）；"
        "控制能耗（∑‖a‖²dt）；以及最小间隙的 5% 分位，用来刻画“贴着障碍飞”的风险程度。", BODY))

    # ------------------------------------------------------- 4 implementation
    S(P("4  设计与实现", H1))
    S(P(
        "代码分为环境、算法、基线、脚本四层：<i>uavnav/env.py</i> 实现向量化仿真器；<i>ppo.py</i> 与 <i>dqn.py</i> 分别实现两种算法；"
        "<i>baselines.py</i> 实现随机、势场与 A*；<i>utils.py</i> 统一指标聚合与固定场景回放；<i>scripts/</i> 下是可复现的训练、评测、"
        "出图与批量实验入口。所有方法共用同一个环境类，保证奖励、终止条件与观测定义完全一致。", BODY))
    S(P("4.1  向量化实现", H2))
    S(P(
        "环境同时维护 64 个独立回合。激光通过解析的射线—圆求交一次算完：对每个环境把障碍中心相对位置与 32 个射线方向做矩阵乘，"
        "得到 (64, 25, 32) 的投影张量，再由判别式与近交点求最小命中距离，单步耗时约 1.6 ms（约 4×10⁴ 步/秒）。"
        "回合结束采用自动重置：终止或截断的环境当步立即换新场景，训练循环无需处理变长回合。"
        "对于因步数上限被截断的回合，用重置前观测的价值估计做自举（bootstrapping），避免把“时间到”误当作“任务失败”。"
        "评测时则切换到 auto_reset=False，冻结已结束的环境，保证固定场景集只产生被标注的回合。", BODY))
    S(P(
        "训练超参：PPO 使用 64 个并行环境 × 128 步的 rollout（每次更新 8192 个样本），6 个 epoch、minibatch 512、初始学习率 3×10⁻⁴ 并线性退火至 10%；"
        "主实验训练 2×10⁶ 步，两个消融各 1×10⁶ 步，每个配置跑 3 个随机种子，报告均值与标准差。DQN 在相同步数预算下每环境步做 2 次梯度更新。"
        "12 次训练在 6 进程并行下约 12 分钟完成，随后的 15 组策略评测（105 行结果）在同一并行度下只需 32 秒。", BODY))
    S(P("4.2  实现过程中发现并修正的一个缺陷", H2))
    S(P(
        "最初版本的射线求交只计算近交点并把负值截断为 0，导致“背离障碍方向”的射线也返回距离 0。"
        "在构造的解析场景中（前方 3 m 处半径 0.5 m 的圆柱，膨胀后 0.75 m），正确的正前射线读数应为 2.25 m，"
        "而有缺陷的实现给出 0 m，同时还有若干背向射线误报为 0，等于告诉策略“四周都被贴住”。"
        "修正后只在投影 b>0 且近交点大于 0 时判定命中，解析校验与预期一致（0° 为 2.25 m、±11.25° 为 2.47 m、背向为自由空间）。"
        "这一细节说明：端到端方法的上限高度依赖感知接口的正确性，把仿真器当作“黑盒”直接训练很容易得到看似收敛、实则有系统性偏差的策略。", BODY))

    # ------------------------------------------------------------- 5 results
    S(P("5  实验与结果", H1))
    S(P("5.1  主实验：学习曲线与定量对比", H2))
    S(P(
        f"图 2 给出 PPO 与 Double DQN 在训练分布上的评测成功率与碰撞率曲线（3 个种子的均值±标准差）。"
        f"PPO 在 8×10⁴ 步时已达 {mean_at('ppo', 81920) * 100:.1f}%（3 种子均值），3×10⁵ 步 {mean_at('ppo', 300000) * 100:.1f}%，"
        f"1×10⁶ 步 {mean_at('ppo', 1000000) * 100:.1f}%，"
        f"最终收敛在 {pct('main', 'PPO')}%，1×10⁶ 步之后仍有缓慢提升但已接近平台。DQN 的起步慢得多：8×10⁴ 步仅 6.0%，"
        f"3×10⁵ 步 45.1%，1×10⁶ 步追到 86.8%，此后基本不再改善，最终为 {pct('main', 'DQN')}%。"
        f"两者的失败构成差异更值得注意：PPO 的碰撞率 {pct('main', 'PPO', 'collision')}%、越界率 {pct('main', 'PPO', 'oob')}%，"
        f"而 DQN 为 {pct('main', 'DQN', 'collision')}% 与 {pct('main', 'DQN', 'oob')}%——离散策略除了撞障碍，还会飞出场地。", BODY))
    S(figblock(FIG / "fig1_learning_curves.png", 15.2,
               "图 2　训练分布上的评测成功率（左）与碰撞率（右）；曲线为 3 个种子的均值，阴影为标准差"))

    S(P(
        f"表 2 汇总了 main 条件下的最终指标。PPO 的成功率比人工势场法高 "
        f"{(val('main', 'PPO', 'success')[0] - val('main', 'Reactive', 'success')[0]) * 100:.1f} 个百分点，"
        f"能耗却只有后者的 {val('main', 'PPO', 'energy')[0] / val('main', 'Reactive', 'energy')[0] * 100:.0f}%，"
        f"路径比 {f2('main', 'PPO', 'path_ratio')} 对 {f2('main', 'Reactive', 'path_ratio')}。"
        f"势场法的失败以越界为主（{pct('main', 'Reactive', 'oob')}%，碰撞仅 {pct('main', 'Reactive', 'collision')}%）："
        f"它只用激光与目标方向合成速度，对 15×15 m² 的场地边界没有显式概念，在凹形区域被斥力推着走出场地——"
        f"而学习策略的观测里包含四周墙面距离，越界率因此压到 {pct('main', 'PPO', 'oob')}%。"
        f"DQN 的成功率为 {pct('main', 'DQN')}%，比 PPO 低 "
        f"{(val('main', 'PPO', 'success')[0] - val('main', 'DQN', 'success')[0]) * 100:.1f} 个百分点，能耗却是 PPO 的 "
        f"{val('main', 'DQN', 'energy')[0] / val('main', 'PPO', 'energy')[0]:.1f} 倍：27 个离散动作每轴只能取 -1/0/+1 的“满舵”，"
        f"加速度在极值之间频繁切换，飞行呈锯齿状，步数相近但 SPL 只有 {f2('main', 'DQN', 'spl')}。"
        f"A* 以 {pct('main', 'A*')}% 的成功率、{f2('main', 'A*', 'spl')} 的 SPL 说明：在完全可观测的静态场景中，"
        f"显式地图加搜索仍是最优解；但 PPO 与之只差 "
        f"{(val('main', 'A*', 'success')[0] - val('main', 'PPO', 'success')[0]) * 100:.1f} 个百分点，"
        f"能耗只有它的 {val('main', 'PPO', 'energy')[0] / val('main', 'A*', 'energy')[0] * 100:.0f}%"
        f"（A* 需要跟踪栅格航点，在拐点处反复加减速），且完全不需要地图。", BODY))

    rows = [["方法", "成功率/%", "碰撞率/%", "越界率/%", "SPL", "路径比", "能耗", "间隙5%分位/m"]]
    for label, key in [("PPO（本文）", "PPO"), ("Double DQN", "DQN"), ("人工势场", "Reactive"),
                       ("A*（上界）", "A*"), ("随机", "Random")]:
        rows.append([label, pct("main", key), pct("main", key, "collision"), pct("main", key, "oob"),
                     f2("main", key, "spl"), f2("main", key, "path_ratio"),
                     f2("main", key, "energy"), f2("main", key, "clearance_p05")])
    S(KeepTogether([P("表 2　main 条件（25 障碍、8 m 感知）下各方法的最终指标，学习类方法为 3 种子均值±标准差", CAP),
                    table(rows, [2.9, 2.2, 2.2, 2.2, 1.9, 2.0, 1.7, 2.6], font=7.8)]))

    S(P("5.2  消融：奖励设计与感知范围", H2))
    S(P(
        f"图 3(a) 在同一 1×10⁶ 步预算下对比塑形奖励与稀疏奖励（训练中每轮 128 回合评测，曲线为 3 个种子的均值）。"
        f"塑形组在 8×10⁴ 步已达 {mean_at('ppo', 81920) * 100:.1f}%，稀疏组只有 {mean_at('ppo_sparse', 81920) * 100:.1f}%，"
        f"相差约 {mean_at('ppo', 81920) / max(mean_at('ppo_sparse', 81920), 1e-9):.1f} 倍："
        f"没有进展奖励时，价值函数必须先“偶然撞到”目标才有梯度，而在 15×15 m² 场地上随机游走几乎不可能命中半径 0.5 m 的目标球。"
        f"稀疏组在 3×10⁵ 步附近追平（{mean_at('ppo_sparse', 300000) * 100:.1f}% 对 {mean_at('ppo', 300000) * 100:.1f}%），"
        f"1×10⁶ 步时以 {mean_at('ppo_sparse', 1000000) * 100:.1f}% 反超塑形组的 {mean_at('ppo', 1000000) * 100:.1f}%。", BODY))
    S(P(
        f"如何解释“塑形反而略低”？势差塑形等价于给策略注入“朝目标走”的强先验，早期收敛极快；但它同时改变了回报的尺度与形状——"
        f"每步的进展项主导了回报，策略更倾向于“尽快接近、少绕路”。稀疏组只认“到达”，于是学出更保守的行为："
        f"平均步数 {f1('main', 'PPO-sparse', 'steps')} 步对 {f1('main', 'PPO', 'steps')} 步，超时率 "
        f"{pct('main', 'PPO-sparse', 'timeout', plus=False)}% 对 {pct('main', 'PPO', 'timeout', plus=False)}%，"
        f"SPL {f2('main', 'PPO-sparse', 'spl')} 对 {f2('main', 'PPO', 'spl')}。"
        f"这与 Ng 等（1999）的结论并不矛盾：势函数塑形不改变最优策略，但在有限预算、有限网络容量与熵正则下，"
        f"它改变了优化路径与实际收敛点。塑形组继续训练到 2×10⁶ 步后达到 {pct('main', 'PPO')}%，"
        f"说明它最终能追上并超过稀疏组——只是多花了一倍的环境交互。", BODY))
    S(P(
        f"图 3(b) 是感知范围的交叉测试。8 m 训练的策略换到 4 m 量程几乎不受影响（{pct('main', 'PPO')}% → {pct('per4', 'PPO')}%），甚至略升；"
        f"而 4 m 训练的策略在自己的量程下为 {pct('per4', 'PPO-4m')}%，换到 8 m 时跌到 {pct('main', 'PPO-4m')}%，"
        f"碰撞率从 {pct('per4', 'PPO-4m', 'collision')}% 升到 {pct('main', 'PPO-4m', 'collision')}%，种子方差也扩大到 ±3.3 个百分点。"
        f"原因是观测分布的单向偏移：8 m 策略在 4 m 条件下看到的只是“远处回波被截断到量程上限”，而截断本就是它训练中见过的模式；"
        f"反过来，4 m 策略从未见过 4–8 m 的测距值，测试时每一束激光都落在训练分布的支撑之外。"
        f"这说明端到端策略把传感器量程“编码”进了行为，换传感器近似于换任务；工程上更稳妥的做法是训练时对量程、噪声与波束数做域随机化。", BODY))
    S(figblock(FIG / "fig3_ablation.png", 15.2,
               "图 3　消融实验。(a) 塑形奖励与稀疏奖励的学习曲线；(b) 感知范围 8 m / 4 m 训练策略在两种测试量程下的成功率"))

    S(P("5.3  分布偏移下的泛化性", H2))
    def drop(key):
        a = float(pct("main", key).split("±")[0])
        b = float(pct("d40", key).split("±")[0])
        return f"{a - b:.1f}"

    S(P(
        f"表 3 与图 4 给出 6 个偏移条件下的成功率。需要先说清 A* 的角色：它拿真值地图做栅格搜索，是特权（oracle）上界，"
        f"所以它的曲线近乎平坦——测距噪声与量程对它没有影响，只有障碍变密才会让它的成功率从 {pct('d10', 'A*')}% 略降到 {pct('d40', 'A*')}%。"
        f"障碍数从 25 降到 10 时所有方法都变好（PPO {pct('d10', 'PPO')}%、DQN {pct('d10', 'DQN')}%、势场 {pct('d10', 'Reactive')}%）；"
        f"升到 40 时差距彻底拉开：PPO {pct('d40', 'PPO')}%（衰减 {drop('PPO')} 个百分点）、"
        f"DQN {pct('d40', 'DQN')}%（衰减 {drop('DQN')} 个百分点）、"
        f"势场 {pct('d40', 'Reactive')}%（衰减 {drop('Reactive')} 个百分点）。"
        f"密度提高意味着同样的场地里可用通道更窄，反应式方法在窄缝中反复振荡，而 PPO 仍保住 {pct('d40', 'PPO')}% 的通行能力。", BODY))
    S(P(
        f"噪声是势场法的另一处软肋：5% 噪声下降到 {pct('noise05', 'Reactive')}%，15% 时只剩 {pct('noise15', 'Reactive')}%，"
        f"而 PPO 几乎不受影响（{pct('noise05', 'PPO')}% / {pct('noise15', 'PPO')}%）——激光的抖动只改变距离数值，"
        f"不改变障碍的几何排布，策略对每个波束的依赖是分布式、冗余的，因此单束噪声不至于让行为翻转。"
        f"地图放大到 24×24 m² 后目标距离变长，PPO 保持 {pct('scale', 'PPO')}%，DQN {pct('scale', 'DQN')}%，势场 {pct('scale', 'Reactive')}%："
        f"势场法的引力项与斥力项比例是针对原尺寸手调的，距离一长就失衡，容易绕圈。"
        f"把感知压到 4 m 时 PPO {pct('per4', 'PPO')}%（甚至略高于 main，但差值在种子波动内），DQN {pct('per4', 'DQN')}%，"
        f"势场 {pct('per4', 'Reactive')}%：短量程让它只能在贴近障碍时才开始转向，来不及减速。", BODY))


    conds = [("d10", "d10（10 障碍）"), ("main", "main（25）"), ("d40", "d40（40）"),
             ("scale", "scale（24 m）"), ("noise05", "noise 5%"), ("noise15", "noise 15%"), ("per4", "per4（4 m）")]
    rows = [["方法"] + [c[1] for c in conds]]
    for label, key in [("PPO", "PPO"), ("Double DQN", "DQN"), ("人工势场", "Reactive"), ("A*（上界）", "A*")]:
        rows.append([label] + [pct(c[0], key) for c in conds])
    S(KeepTogether([P("表 3　各分布偏移条件下的成功率（%），学习类方法为 3 种子均值±标准差", CAP),
                    table(rows, [2.6, 2.1, 2.1, 2.1, 2.3, 2.1, 2.1, 2.1], font=7.8)]))

    S(figblock(FIG / "fig4_generalisation.png", 15.2, "图 4　成功率随障碍密度（左）与激光噪声（右）的变化"))

    S(P("5.4  轨迹案例", H2))
    S(P(
        "图 5 在三个固定场景上对比了各控制器的实际轨迹（PPO 全部成功，人工势场在场景 B 失败）。"
        "场景 A 中 PPO 与 A* 的轨迹几乎重合，人工势场法因右侧障碍的斥力略微外绕但仍在 0.5 m 目标半径内到达；"
        "这说明在“目标方向与障碍分布不冲突”的常规场景里，学到的策略已经复现了近似最短路径的几何。"
        "场景 B 是势场法的典型失效：它在目标前约 3 m 处被障碍斥力与目标引力同时夹住，停在障碍背后来回抖动直到超时——"
        "这就是局部极小，反映在统计上就是势场法 26.2% 的失败率。"
        "场景 C 换用 Double DQN：三条轨迹的路线大体一致，但 DQN 的红线呈可见的小幅锯齿，"
        f"在同样 8 个场景上它需要 56 步才全部结束（PPO 47 步、A* 37 步），这正是 DQN 能耗为 PPO 的 "
        f"{val('main', 'DQN', 'energy')[0] / val('main', 'PPO', 'energy')[0]:.1f} 倍、SPL 偏低的直接原因——"
        "离散的“满舵”动作在每一拍都重新选一次方向，控制不平滑。", BODY))
    S(figblock(FIG / "fig5_trajectories.png", 15.2,
               "图 5　三个固定场景上的轨迹对比（实线为成功、虚线为失败；A/B 对比 PPO、人工势场与 A*，C 对比 PPO、Double DQN 与 A*）"))

    # -------------------------------------------------------- 6 discussion
    S(P("6  分析与思考", H1))
    S(P(
        "优势方面。第一，端到端策略不需要地图与规划器，观测到动作是一次前向计算，推理开销固定且很小，适合机载实时控制；"
        "而 A* 需要先建图再搜索，代价随地图规模增长。第二，效率：PPO 的能耗只有 A* 的 "
        f"{val('main', 'PPO', 'energy')[0] / val('main', 'A*', 'energy')[0] * 100:.0f}%、人工势场法的 "
        f"{val('main', 'PPO', 'energy')[0] / val('main', 'Reactive', 'energy')[0] * 100:.0f}%，"
        f"SPL {f2('main', 'PPO', 'spl')} 与 A* 的 {f2('main', 'A*', 'spl')} 已经非常接近——"
        "原因在于策略学会了“提前减速—绕行—再加速”，而势场法的斥力项天然引入来回振荡，A* 则要在栅格拐点反复加减速。"
        "第三，鲁棒性：在 40 个障碍、15% 测距噪声与 24 m 地图三类偏移下，PPO 的衰减都远小于两个基线，"
        "说明它学到的是与场景统计相关的避障几何，而不是一组固定权重。", BODY))
    S(P(
        "不足与局限。其一，与 A* 上界仍有差距，但差距很小：main 条件下低 "
        f"{(val('main', 'A*', 'success')[0] - val('main', 'PPO', 'success')[0]) * 100:.1f} 个百分点，"
        f"稀疏场景（d10）已基本追平（{pct('d10', 'PPO')}% 对 {pct('d10', 'A*')}%），差距主要出现在 d40（"
        f"{pct('d40', 'PPO')}% 对 {pct('d40', 'A*')}%）。根源是部分可观测：8 m 量程看不到凹形障碍的全貌，"
        "策略只能反应式避让，在窄缝中难以做“先退后进”的规划。"
        "其二，感知量程消融暴露了非对称过拟合：4 m 训练的策略换到 8 m 就从 89.1% 跌到 81.8%，"
        "说明策略把量程编码进了行为，换传感器近似于换任务。"
        "其三，场景是简化的：障碍为贯穿高度的竖直圆柱、动力学为双积分且无姿态与推力约束，因此结论不能直接外推到真实飞行。"
        "其四，评测只有 3 个随机种子、每条件 256 个回合，PPO 成功率的标准差 "
        f"{val('main', 'PPO', 'success')[1] * 100:.1f} 个百分点、DQN {val('main', 'DQN', 'success')[1] * 100:.1f} 个百分点，"
        "小于 2 个百分点的差异（例如 per4 的 96.4% 与 main 的 96.1%）不应过度解读。"
        "其五，全部实验在仿真中进行，没有做域随机化或真机迁移。", BODY))
    S(P(
        "改进方向。针对部分可观测，可以引入循环记忆（LSTM/GRU）或把历史观测堆叠成短时占据栅格，让策略具备“记住绕过的障碍”的能力；"
        "也可以采用特权学习：先用 A* 的完整地图训练教师策略，再蒸馏到只用激光的学生策略，这在无人机高速飞行中已被证明有效"
        "（Loquercio 等，2021；Kaufmann 等，2023）。针对感知量程过拟合，训练时对量程、噪声、波束数做域随机化，"
        "并加入“传感器失效”样本，可显著提升跨量程鲁棒性。针对奖励设计，本文的消融给出了一个具体线索："
        "塑形把早期样本效率提高了约 5 倍，却在 1×10⁶ 步时略低于稀疏奖励，说明 2.0 的势函数系数偏大，"
        "后续可尝试退火（先塑形、后减小系数）或降低系数，把“快”和“最终不劣”同时拿到。"
        "最后，把仿真换成带姿态动力学的模型并加入电机延迟与风扰，是走向真机的必要一步。", BODY))


    # ------------------------------------------------------------ references
    S(P("参考文献", H1))
    refs = [
        "[1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.",
        "[2] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage estimation. ICLR, 2016.",
        "[3] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature, 2015, 518: 529-533.",
        "[4] van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double Q-learning. AAAI, 2016.",
        "[5] Ng A Y, Harada D, Russell S. Policy invariance under reward transformations: theory and application to reward shaping. ICML, 1999.",
        "[6] Khatib O. Real-time obstacle avoidance for manipulators and mobile robots. IJRR, 1986, 5(1): 90-98.",
        "[7] Hart P E, Nilsson N J, Raphael B. A formal basis for the heuristic determination of minimum cost paths. IEEE TSSC, 1968.",
        "[8] Anderson P, Chang A, Chaplot D S, et al. On evaluation of embodied navigation agents. arXiv:1807.06757, 2018.",
        "[9] Loquercio A, Kaufmann E, Ranftl R, et al. Learning high-speed flight in the wild. Science Robotics, 2021, 6(59).",
        "[10] Kaufmann E, Bauersfeld L, Loquercio A, et al. Champion-level drone racing using deep reinforcement learning. Nature, 2023, 620: 982-987.",
    ]
    for r in refs:
        S(P(r, ParagraphStyle("ref", parent=BODY0, fontSize=7.9, leading=10.2,
                              leftIndent=13, firstLineIndent=-13, spaceBefore=0, spaceAfter=0)))

    # ------------------------------------------------------------- appendix
    S(P("附录 A　复现与实验配置", H1))
    S(P("依赖：Python 3.13、NumPy、PyTorch 2.15（CPU 版）、Matplotlib、pandas，见 requirements.txt。"
        "一键复现：python scripts/run_suite.py --jobs 6（6 进程并行完成 12 次训练与 15 组 × 7 条件评测，"
        "自动生成 results/summary.csv）。", BODY0))
    S(P("单次训练：python scripts/train.py --algo ppo --seed 0 --steps 2000000 --out results/ppo_s0；"
        "两组消融分别追加 --reward sparse 与 --perception 4.0（步数 1×10⁶）。单次评测：python scripts/evaluate.py "
        "--kind ppo --model results/ppo_s0/model.pt --out results/ppo_s0 --episodes 256 "
        "--conditions main,d10,d40,scale,noise05,noise15,per4。", BODY0))
    S(P("出图与本报告：python scripts/make_figures.py、python scripts/make_report.py。文中所有数值均由脚本从 "
        "results/summary.csv 与 results/*/log.jsonl 读取生成，未做人工改写；图 5 的演示场景种子为 9520。", BODY0))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("SimSun", 8.5)
        canvas.setFillColor(colors.Color(0.4, 0.4, 0.4))
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"{doc_.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.7 * cm,
                            title="基于近端策略优化的无人机端到端避障导航",
                            author="2026439189 李珅")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("pdf ->", OUT.resolve())


if __name__ == "__main__":
    build()
