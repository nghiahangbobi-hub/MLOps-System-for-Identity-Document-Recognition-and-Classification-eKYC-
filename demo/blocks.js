function blockBar(ratio) {
  const value = Math.min(1, Math.max(0, ratio));
  const filled = value === 1 ? 10 : Math.min(9, Math.round(value * 10));
  const percent = new Intl.NumberFormat('vi', {style: 'percent', maximumFractionDigits: 1});
  return `[${'█'.repeat(filled)}${'░'.repeat(10 - filled)}] ${percent.format(value)}`;
}
