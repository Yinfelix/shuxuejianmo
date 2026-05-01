import {
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Stat,
  Stack,
  Table,
  Text,
  BarChart,
} from "cursor/canvas";

const COMPARE_K3 = [
  ["指标", "基础巡检", "联合巡检", "变化"],
  ["空中完成时间 (s)", "361.2", "570.9", "-58.0%"],
  ["地面复核旅行时间 (s)", "1 268.0", "1 109.0", "+12.5%"],
  ["地面服务时间 (s)", "2 940.0", "2 190.0", "+25.5%"],
  ["总闭环时间 (s)", "4 569.2", "3 869.9", "+15.3%"],
  ["无人机直接确认数", "0/16", "5/16", "-"],
  ["人工复核点数", "16/16", "11/16", "-31.3%"],
  ["人工复核比例", "100%", "68.8%", "-31.2pp"],
];

const COMPARE_K4 = [
  ["指标", "基础巡检", "联合巡检", "变化"],
  ["空中完成时间 (s)", "322.5", "562.6", "-74.5%"],
  ["地面复核旅行时间 (s)", "1 268.0", "1 089.0", "+14.1%"],
  ["地面服务时间 (s)", "2 940.0", "1 890.0", "+35.7%"],
  ["总闭环时间 (s)", "4 530.5", "3 541.6", "+21.8%"],
  ["无人机直接确认数", "0/16", "7/16", "-"],
  ["人工复核点数", "16/16", "9/16", "-43.8%"],
  ["人工复核比例", "100%", "56.3%", "-43.7pp"],
];

const K_SUMMARY = [
  ["K", "空中完成 (s)", "地面复核 (s)", "总闭环 (s)", "直接确认", "确认比例", "人工复核"],
  ["K=1", "1 312", "4 208", "5 520", "0/16", "0%", "16"],
  ["K=2", "537", "4 208", "4 745", "0/16", "0%", "16"],
  ["K=3", "571", "3 299", "3 870", "5/16", "31.3%", "11"],
  ["K=4", "563", "2 979", "3 542", "7/16", "43.8%", "9"],
];

const NODE_TABLE = [
  ["#", "点位名称", "优先级", "基准悬停", "确认阈值", "GA分配", "需额外", "人工服务", "直接确认"],
  ["1", "B1-东立面裂缝", "★★", "50s", "200s", "200s", "0s", "180s", "是"],
  ["2", "B1-屋顶积水", "★★★", "60s", "520s", "60s", "460s", "180s", "否"],
  ["3", "B1-消防连廊顶部", "★", "35s", "110s", "110s", "0s", "120s", "是"],
  ["4", "B2-空调支架锈蚀", "★★", "55s", "240s", "55s", "185s", "180s", "否"],
  ["5", "B2-南立面渗水点", "★★★", "50s", "420s", "50s", "370s", "180s", "否"],
  ["6", "B2-女儿墙破损", "★★", "40s", "180s", "40s", "140s", "150s", "否"],
  ["7", "B3-女儿墙贯裂缝", "★★", "45s", "200s", "45s", "155s", "150s", "否"],
  ["8", "B3-南立面贯穿裂缝", "★★★", "60s", "560s", "60s", "500s", "180s", "否"],
  ["9", "B4-女儿墙裂缝", "★★★", "55s", "520s", "55s", "465s", "180s", "否"],
  ["10", "B4-南立面积水病害", "★★", "45s", "190s", "190s", "0s", "180s", "是"],
  ["11", "B5-南立面标识牌节点", "★", "35s", "110s", "110s", "0s", "120s", "是"],
  ["12", "B5-女儿墙变形缝", "★★", "40s", "170s", "170s", "0s", "150s", "是"],
  ["13", "B6-南立面地基异常", "★★★", "60s", "540s", "60s", "480s", "180s", "否"],
  ["14", "B6-女儿墙渗水裂缝", "★★", "50s", "330s", "50s", "280s", "180s", "否"],
  ["15", "B7-女儿墙贯裂缝", "★★", "45s", "360s", "45s", "315s", "180s", "否"],
  ["16", "B7-南立面围栏松动", "★★★", "65s", "560s", "65s", "495s", "180s", "否"],
];

const SENSITIVITY_TABLE = [
  ["阈值缩放", "阈值范围 (s)", "可确认节点", "仍需人工", "节省人工比例"],
  ["×0.5", "55 – 280", "16", "0", "100%"],
  ["×0.7", "77 – 392", "16", "0", "100%"],
  ["×0.8", "88 – 448", "16", "0", "100%"],
  ["×1.0 (基准)", "110 – 560", "16", "0", "100%"],
  ["×1.2", "132 – 672", "11", "5", "68.8%"],
  ["×1.4", "154 – 784", "10", "6", "62.5%"],
];

const ENERGY_TABLE = [
  ["#", "点位名称", "阈值 (s)", "单次最大悬停 (s)", "飞行能耗 (J)", "能量约束瓶颈?", "状态"],
  ["1", "B1-东立面裂缝", "200", "584.6", "6 380", "否", "可直接确认"],
  ["2", "B1-屋顶积水", "520", "584.9", "6 317", "否", "阈值超限"],
  ["3", "B1-消防连廊顶部", "110", "587.1", "5 838", "否", "可直接确认"],
  ["4", "B2-空调支架锈蚀", "240", "584.7", "6 376", "否", "阈值超限"],
  ["5", "B2-南立面渗水点", "420", "583.1", "6 716", "否", "阈值超限"],
  ["6", "B2-女儿墙破损", "180", "584.0", "6 510", "否", "阈值超限"],
  ["7", "B3-女儿墙贯裂缝", "200", "578.8", "7 672", "否", "阈值超限"],
  ["8", "B3-南立面贯穿裂缝", "560", "578.6", "7 704", "是", "Hard Point"],
  ["9", "B4-女儿墙裂缝", "520", "590.5", "5 083", "否", "阈值超限"],
  ["10", "B4-南立面积水病害", "190", "589.5", "5 300", "否", "可直接确认"],
  ["11", "B5-南立面标识牌节点", "110", "583.4", "6 663", "否", "可直接确认"],
  ["12", "B5-女儿墙变形缝", "170", "587.6", "5 732", "否", "可直接确认"],
  ["13", "B6-南立面地基异常", "540", "586.6", "5 954", "否", "阈值超限"],
  ["14", "B6-女儿墙渗水裂缝", "330", "585.0", "6 308", "否", "阈值超限"],
  ["15", "B7-女儿墙贯裂缝", "360", "571.1", "9 366", "是", "Hard Point"],
  ["16", "B7-南立面围栏松动", "560", "569.2", "9 786", "是", "Hard Point"],
];

const RECOMMENDATIONS = [
  {
    category: "无人机数量",
    suggestion: "推荐 K=4",
    rationale:
      "总闭环时间最短（3 542s），人工复核点数最少（9 个），直接确认比例最高（43.8%）。增加一架无人机后，地面复核工作量减少 18%，整体效率显著提升。",
    caution: "K=5 及以上需评估额外硬件与调度成本",
  },
  {
    category: "直接确认阈值",
    suggestion: "阈值设定上限约 600s",
    rationale:
      "各点位单次最大悬停容量为 569-591s（K=3），阈值 ≤200s 的 5 个点位可直接确认。积水类病害（520-560s）即使在能量约束内也无法单次确认，应保持人工复核路径。",
    caution: "阈值 >580s 时会形成 Hard Point，建议不超过单次最大悬停上限",
  },
  {
    category: "悬停时间预算",
    suggestion: "单架额外悬停预算 ≤600s",
    rationale:
      "K=3 额外悬停总量 575s，K=4 额外悬停总量 870s。建议单次出击悬停总时长不超过 600s，确保时间窗口（2600s）和能量（135 kJ）双约束均满足。",
    caution: "需在航线规划中预留返航能量冗余（建议 ≥5%）",
  },
];

export default function AnalysisDashboard() {
  return (
    <Stack gap={24}>
      <Stack gap={4}>
        <H1>联合巡检方案 — 结果分析报告</H1>
        <Text tone="secondary" size="small">
          GA 优化 · 种子 11 · 16 目标点位 · K=3 &amp; K=4 · Horizon=2600s · 能量=135 kJ
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="15.3%" label="K=3 总时间节省" tone="success" />
        <Stat value="21.8%" label="K=4 总时间节省" tone="success" />
        <Stat value="31.3%" label="K=3 人工复核减少" tone="success" />
        <Stat value="43.8%" label="K=4 直接确认比例" tone="success" />
      </Grid>

      <Divider />

      <Stack gap={12}>
        <H2>(1) 联合巡检 vs 基础巡检</H2>
        <Table
          headers={COMPARE_K3[0]}
          rows={COMPARE_K3.slice(1)}
          rowTone={[
            undefined, undefined, undefined,
            "success", undefined, "success",
            "success", undefined, "success",
          ]}
        />
        <Table
          headers={COMPARE_K4[0]}
          rows={COMPARE_K4.slice(1)}
          rowTone={[
            undefined, undefined, undefined,
            "success", undefined, "success",
            "success", undefined, "success",
          ]}
        />
        <Text size="small" tone="secondary">
          联合方案通过无人机在低阈值点位直接悬停确认，替代人工爬楼，将地面服务时间减少 25-36%，推动总闭环时间缩短 15-22%。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>(2) 不同 K 值的指标变化</H2>
        <BarChart
          data={[
            { label: "K=1", values: [5520, 16, 0] },
            { label: "K=2", values: [4745, 16, 0] },
            { label: "K=3", values: [3870, 11, 31] },
            { label: "K=4", values: [3542, 9, 44] },
          ]}
          series={[
            { label: "总闭环时间 (s)", tone: "accent" },
            { label: "人工复核点数", tone: "secondary" },
            { label: "直接确认比例 (%)", tone: "success" },
          ]}
        />
        <Table headers={K_SUMMARY[0]} rows={K_SUMMARY.slice(1)} />
        <Text size="small" tone="secondary">
          K≥3 时无人机具备直接确认能力。K=4 相比 K=3：总时间再缩短 328s（-8.5%），人工复核再减少 2 点，直接确认比例提升 12.5pp。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>(3) 直接确认阈值敏感度分析 (K=3)</H2>
        <Table
          headers={SENSITIVITY_TABLE[0]}
          rows={SENSITIVITY_TABLE.slice(1)}
          rowTone={[
            undefined, undefined, undefined,
            "success", undefined, "success",
            undefined, undefined, "warning",
            undefined, undefined, "warning",
          ]}
        />
        <Text size="small" tone="secondary">
          阈值缩放 ×1.2 是拐点：超过此系数后，B3/B7/B8/B9/B15/B16 共 6 个高阈值积水类病害将转为 Hard Point（单次出击无法覆盖）。建议阈值不超过基准值的 1.1 倍。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>(3b) 各点位能量约束详情 (K=3)</H2>
        <Table
          headers={ENERGY_TABLE[0]}
          rows={ENERGY_TABLE.slice(1)}
          rowTone={[
            "success", undefined, undefined, undefined, undefined, undefined, "success",
            "warning", undefined, undefined, undefined, undefined, "warning", "warning",
            "success", undefined, undefined, undefined, undefined, undefined, "success",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            "warning", undefined, undefined, undefined, undefined, "warning", "warning",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            "success", undefined, undefined, undefined, undefined, undefined, "success",
            "success", undefined, undefined, undefined, undefined, undefined, "success",
            "success", undefined, undefined, undefined, undefined, undefined, "success",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            undefined, undefined, undefined, undefined, undefined, undefined, "warning",
            "warning", undefined, undefined, undefined, undefined, "warning", "warning",
            "warning", undefined, undefined, undefined, undefined, "warning", "warning",
          ]}
        />
        <Text size="small" tone="secondary">
          节点 8、15、16 为能量瓶颈（Hard Point），B7 区飞行能耗高达 9 366-9 786 J，限制了悬停容量。单次最大悬停均约 569-591s，是确认阈值的硬上限。
        </Text>
      </Stack>

      <Stack gap={12}>
        <H2>(3c) 各点位 GA 确认决策</H2>
        <Table
          headers={NODE_TABLE[0]}
          rows={NODE_TABLE.slice(1)}
          rowTone={[
            "success", "default", "success",
            "warning", "default", "default",
            "success", "default", "success",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
            "success", "default", "success",
            "success", "default", "success",
            "success", "default", "success",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
            "warning", "default", "default",
          ]}
        />
        <Text size="small" tone="secondary">
          GA 选取节点 1, 3, 10, 11, 12 直接确认，共同特征：阈值 ≤200s，额外悬停 0-150s，在单次最大悬停容量内可直接覆盖。剩余 11 个节点因阈值过高需人工复核。
        </Text>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>(4) 配置与阈值推荐建议</H2>
        <Grid columns={3} gap={16}>
          {RECOMMENDATIONS.map((r) => (
            <Stack key={r.category} gap={8}>
              <H3>{r.category}</H3>
              <Text size="small" tone="accent">
                {r.suggestion}
              </Text>
              <Text size="small">{r.rationale}</Text>
              <Divider />
              <Text size="small" tone="warning">
                {r.caution}
              </Text>
            </Stack>
          ))}
        </Grid>
      </Stack>

      <Divider />

      <Text tone="secondary" size="small">
        数据来源：GA 优化解 (种子 11) · 问题数据 Horizon=2600s / 能量=135 kJ / 悬停功率=220 J/s · 敏感度分析为直接计算（非完整重跑 GA）
      </Text>
    </Stack>
  );
}
