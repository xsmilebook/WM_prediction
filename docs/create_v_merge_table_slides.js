const fs = require('fs');
const path = require('path');

const PptxGenJS = require('pptxgenjs');
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require('/home/cuizaixu_lab/xuhaoshu/.codex/skills/slides/assets/pptxgenjs_helpers/layout');

const DATA_ROOT = path.resolve(__dirname, '..', '..', 'data');
const OUTPUT_PPTX = path.join(__dirname, 'v_merge_table_slides.pptx');

const FEATURE_COLUMNS = [
  { key: 'GGFC', label: 'GGFC', merged: false },
  { key: 'GWFC', label: 'GWFC', merged: false },
  { key: 'WWFC', label: 'WWFC', merged: false },
  { key: 'GG_GW_MergedFC', label: 'GG+GW', merged: true },
  { key: 'GG_WW_MergedFC', label: 'GG+WW', merged: true },
  { key: 'GW_WW_MergedFC', label: 'GW+WW', merged: true },
  { key: 'GG_GW_WW_MergedFC', label: 'GG+GW+WW', merged: true },
];

const AGE_FILES = [
  { label: 'HCPD', file: path.join(DATA_ROOT, 'HCPD', 'prediction', 'feature_merge_summary_age.csv') },
  { label: 'PNC', file: path.join(DATA_ROOT, 'PNC', 'prediction', 'feature_merge_summary_age.csv') },
  { label: 'CCNP', file: path.join(DATA_ROOT, 'CCNP', 'prediction', 'feature_merge_summary_age.csv') },
  { label: 'EFNY', file: path.join(DATA_ROOT, 'EFNY', 'prediction', 'feature_merge_summary_age.csv') },
];

const COGNITION_FILE = path.join(DATA_ROOT, 'ABCD', 'prediction', 'feature_merge_summary_cognition.csv');
const PFACTOR_FILE = path.join(DATA_ROOT, 'ABCD', 'prediction', 'feature_merge_summary_pfactor.csv');

const COGNITION_ROWS = [
  { key: 'nihtbx_cryst_uncorrected', label: 'Crystallized' },
  { key: 'nihtbx_fluidcomp_uncorrected', label: 'Fluid' },
  { key: 'nihtbx_totalcomp_uncorrected', label: 'Total' },
];

const PFACTOR_ROWS = [
  { key: 'General', label: 'General' },
  { key: 'Ext', label: 'Ext' },
  { key: 'ADHD', label: 'ADHD' },
  { key: 'Int', label: 'Int' },
];

const COLORS = {
  bg: 'F6F7FB',
  title: '1F2937',
  subtitle: '5B6575',
  border: 'C9D1E1',
  headerFill: 'E8EEF9',
  rowHeaderFill: 'EEF2F8',
  cellFill: 'FFFFFF',
  highlight: 'C62828',
  text: '1F2937',
};

function parseCsv(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8').trim();
  const lines = raw.split(/\r?\n/);
  const header = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row = {};
    header.forEach((key, index) => {
      row[key] = values[index];
    });
    row.median_corr = Number(row.median_corr);
    return row;
  });
}

function buildFeatureMap(rows, rowKey) {
  const featureMap = {};
  for (const row of rows) {
    if (row.target === rowKey) {
      featureMap[row.feature_set] = row.median_corr;
    }
  }
  return featureMap;
}

function buildAgeRows() {
  return AGE_FILES.map(({ label, file }) => {
    const featureMap = buildFeatureMap(parseCsv(file), 'age');
    return { label, featureMap };
  });
}

function buildTargetRows(filePath, rowSpecs) {
  const rows = parseCsv(filePath);
  return rowSpecs.map(({ key, label }) => ({
    label,
    featureMap: buildFeatureMap(rows, key),
  }));
}

function getBestBaseline(featureMap) {
  return Math.max(featureMap.GGFC, featureMap.GWFC, featureMap.WWFC);
}

function addCell(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    shape: { type: 'rect' },
    line: { color: COLORS.border, width: 1 },
    fill: { color: options.fill || COLORS.cellFill },
    margin: 0.05,
    align: options.align || 'center',
    valign: 'mid',
    fontFace: 'Calibri',
    fontSize: options.fontSize || 14,
    bold: Boolean(options.bold),
    color: options.color || COLORS.text,
    breakLine: false,
  });
}

function formatCorr(value) {
  return value.toFixed(3);
}

function drawTable(slide, rows) {
  const left = 0.45;
  const top = 1.38;
  const tableWidth = 12.43;
  const rowHeaderWidth = 2.2;
  const valueWidth = (tableWidth - rowHeaderWidth) / FEATURE_COLUMNS.length;
  const headerHeight = 0.68;
  const rowHeight = rows.length === 3 ? 0.94 : 0.82;

  addCell(slide, '', left, top, rowHeaderWidth, headerHeight, {
    fill: COLORS.headerFill,
    bold: true,
    fontSize: 13,
  });

  FEATURE_COLUMNS.forEach((column, index) => {
    addCell(
      slide,
      column.label,
      left + rowHeaderWidth + index * valueWidth,
      top,
      valueWidth,
      headerHeight,
      { fill: COLORS.headerFill, bold: true, fontSize: 13 }
    );
  });

  rows.forEach((row, rowIndex) => {
    const y = top + headerHeight + rowIndex * rowHeight;
    const bestBaseline = getBestBaseline(row.featureMap);

    addCell(slide, row.label, left, y, rowHeaderWidth, rowHeight, {
      fill: COLORS.rowHeaderFill,
      bold: true,
      fontSize: 14,
    });

    FEATURE_COLUMNS.forEach((column, colIndex) => {
      const value = row.featureMap[column.key];
      const highlighted = column.merged && value > bestBaseline;
      addCell(
        slide,
        formatCorr(value),
        left + rowHeaderWidth + colIndex * valueWidth,
        y,
        valueWidth,
        rowHeight,
        {
          fill: COLORS.cellFill,
          fontSize: 14,
          bold: highlighted,
          color: highlighted ? COLORS.highlight : COLORS.text,
        }
      );
    });
  });
}

function addSlideTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.45,
    y: 0.3,
    w: 8.5,
    h: 0.35,
    fontFace: 'Calibri',
    fontSize: 24,
    bold: true,
    color: COLORS.title,
    margin: 0,
  });

  slide.addText(subtitle, {
    x: 0.45,
    y: 0.72,
    w: 7.9,
    h: 0.25,
    fontFace: 'Calibri',
    fontSize: 11,
    color: COLORS.subtitle,
    margin: 0,
  });

  slide.addText('Red bold: merged feature > best baseline', {
    x: 8.55,
    y: 0.72,
    w: 4.35,
    h: 0.25,
    fontFace: 'Calibri',
    fontSize: 11,
    bold: true,
    color: COLORS.highlight,
    align: 'right',
    margin: 0,
  });
}

function addSlideFootnote(slide, note) {
  slide.addText(note, {
    x: 0.45,
    y: 6.95,
    w: 12.43,
    h: 0.18,
    fontFace: 'Calibri',
    fontSize: 9,
    color: COLORS.subtitle,
    margin: 0,
  });
}

async function main() {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'OpenAI Codex';
  pptx.company = 'WM_prediction';
  pptx.subject = 'V_merge correlation tables';
  pptx.title = 'V_merge result tables';
  pptx.lang = 'en-US';
  pptx.theme = {
    headFontFace: 'Calibri',
    bodyFontFace: 'Calibri',
    lang: 'en-US',
  };

  const slideSpecs = [
    {
      title: 'Age prediction across datasets',
      subtitle: 'Median correlation from feature_merge_summary_age.csv',
      rows: buildAgeRows(),
      note: 'Datasets: HCPD, PNC, CCNP, EFNY',
    },
    {
      title: 'ABCD cognition prediction',
      subtitle: 'Median correlation from feature_merge_summary_cognition.csv',
      rows: buildTargetRows(COGNITION_FILE, COGNITION_ROWS),
      note: 'Targets: nihtbx crystallized, fluid, and total composite scores',
    },
    {
      title: 'ABCD p-factor prediction',
      subtitle: 'Median correlation from feature_merge_summary_pfactor.csv',
      rows: buildTargetRows(PFACTOR_FILE, PFACTOR_ROWS),
      note: 'Targets: General, Ext, ADHD, Int',
    },
  ];

  slideSpecs.forEach((spec) => {
    const slide = pptx.addSlide();
    slide.background = { color: COLORS.bg };
    addSlideTitle(slide, spec.title, spec.subtitle);
    drawTable(slide, spec.rows);
    addSlideFootnote(slide, spec.note);
    warnIfSlideHasOverlaps(slide, pptx);
    warnIfSlideElementsOutOfBounds(slide, pptx);
  });

  await pptx.writeFile({ fileName: OUTPUT_PPTX });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
