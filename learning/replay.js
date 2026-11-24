// Replay the control-loop frames at their simulation timestamps.
class CameraReplay {
  constructor(root) {
    this.root = root;
    this.canvas = root.querySelector("canvas");
    this.controls = root.querySelector(".replay-controls");
    this.seek = root.querySelector('[type="range"]');
    this.overlay = root.querySelector('[type="checkbox"]');
    this.play = root.querySelector("button");
    this.status = root.querySelector("output");
    this.stickPanel = root.querySelector(".stick-panel");
    // Recordings store AETR: roll, pitch, throttle, yaw.
    this.sticks = [["ロール", 0], ["ピッチ", 1], ["ヨー", 3], ["スロットル", 2]].map(([label, channel]) => {
      const meter = document.createElement("div");
      meter.className = "stick-meter";
      meter.setAttribute("role", "meter");
      meter.setAttribute("aria-label", label);
      meter.setAttribute("aria-valuemin", channel === 2 ? "0" : "-100");
      meter.setAttribute("aria-valuemax", "100");
      const value = document.createElement("b");
      const track = document.createElement("span");
      track.className = "stick-track";
      meter.append(label, value, track);
      this.stickPanel.append(meter);
      return { channel, meter, value };
    });
    this.play.onclick = () => {
      if (this.playing) return this.pause();
      this.playing = true;
      this.play.textContent = "一時停止";
      const start = this.index === this.frames.length - 1 ? 0 : this.index;
      this.origin = performance.now() - this.frames[start].t * 1000;
      this.draw(start);
    };
    this.seek.oninput = () => {
      this.pause();
      this.draw(Number(this.seek.value));
    };
    this.overlay.onchange = () => {
      if (this.current) this.paint(this.current, this.frames[this.index]);
    };
    this.clear();
  }
  pause() {
    clearTimeout(this.timer);
    this.playing = false;
    this.play.textContent = "再生";
  }
  clear(message = "") {
    this.pause();
    this.token = {};
    this.request = 0;
    this.cache = new Map();
    this.frames = [];
    this.current = null;
    this.index = 0;
    this.canvas.hidden = this.controls.hidden = this.stickPanel.hidden = true;
    this.status.textContent = message;
  }
  async open(path, trace) {
    this.clear("カメラ映像を読み込み中…");
    const token = this.token;
    try {
      const response = await fetch(path);
      if (!response.ok) throw Error();
      const frames = await response.json();
      if (this.token !== token) return;
      if (!frames.length) throw Error();
      this.path = path;
      this.trace = trace;
      this.frames = frames;
      this.seek.max = frames.length - 1;
      await this.draw(0);
    } catch {
      if (this.token === token) this.clear("カメラ記録を読み込めませんでした。");
    }
  }
  image(index) {
    if (!this.cache.has(index)) {
      const image = new Image();
      image.src = this.path + "/frame-" + index;
      this.cache.set(index, image.decode().then(() => image));
    }
    return this.cache.get(index);
  }
  async draw(index) {
    clearTimeout(this.timer);
    const token = this.token, request = ++this.request;
    try {
      const image = await this.image(index);
      if (token !== this.token || request !== this.request) return;
      this.index = index;
      this.seek.value = index;
      this.current = image;
      this.canvas.hidden = this.controls.hidden = false;
      const frame = this.frames[index];
      this.paint(image, frame);
      this.showSticks(frame.aetr);
      const ids = Object.keys(frame.gates);
      this.status.textContent = frame.t.toFixed(3) + " / " + this.frames.at(-1).t.toFixed(3)
        + " 秒　目標 G" + frame.target + "　検出 " + (ids.map(id => "G" + id).join("・") || "なし")
        + (frame.confirmed_gates != null ? "　推定通過 " + frame.predicted_gates + "（確認済み " + frame.confirmed_gates + "）" : "")
        + "　PnP " + (frame.masked ? "意図的な検出欠落" : frame.pnp ? "成功" : "未取得")
        + (frame.pnp ? "" : "（" + frame.age.toFixed(2) + " 秒）")
        + "　推定高度 " + frame.position[2].toFixed(2) + " m"
        + "　飛行高度 " + this.trace[index][3].toFixed(2) + " m";
      for (const key of this.cache.keys()) {
        if (Math.abs(index - key) > 3) this.cache.delete(key);
      }
      if (index + 1 === this.frames.length) return this.pause();
      this.image(index + 1).catch(() => {});
      if (this.playing) {
        const due = this.origin + 1000 * this.frames[index + 1].t;
        this.timer = setTimeout(() => this.draw(index + 1), Math.max(0, due - performance.now()));
      }
    } catch {
      if (token === this.token && request === this.request) {
        this.pause();
        this.cache.delete(index);
        this.status.textContent = "このフレームを読み込めませんでした。";
      }
    }
  }
  showSticks(aetr) {
    this.stickPanel.hidden = !(Array.isArray(aetr) && aetr.length === 4 && aetr.every(Number.isFinite));
    if (this.stickPanel.hidden) return;
    for (const { channel, meter, value } of this.sticks) {
      const throttle = channel === 2;
      const input = Math.max(-1, Math.min(1, aetr[channel]));
      const percent = Math.round(throttle ? (input + 1) * 50 : input * 100);
      const label = (!throttle && percent > 0 ? "+" : "") + percent + "%";
      value.textContent = label;
      meter.setAttribute("aria-valuenow", percent);
      meter.setAttribute("aria-valuetext", label);
      meter.style.setProperty("--start", (throttle ? 0 : 50 + Math.min(0, percent) / 2) + "%");
      meter.style.setProperty("--fill", (throttle ? percent : Math.abs(percent) / 2) + "%");
    }
  }
  paint(image, frame) {
    if (this.canvas.width !== image.naturalWidth || this.canvas.height !== image.naturalHeight) {
      this.canvas.width = image.naturalWidth;
      this.canvas.height = image.naturalHeight;
    }
    const ctx = this.canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    if (!this.overlay.checked) return;
    for (const [id, points] of Object.entries(frame.gates)) {
      ctx.strokeStyle = ctx.fillStyle = Number(id) === frame.target ? "#51ff9e" : "#ffcc54";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath();
      ctx.stroke();
      for (const [x,y] of points) {
        ctx.beginPath();
        ctx.arc(x,y,2.6,0,Math.PI*2);
        ctx.fill();
      }
      const x = Math.max(1, Math.min(this.canvas.width - 36, points[0][0])), y = Math.max(13, Math.min(this.canvas.height - 13, points[0][1]));
      ctx.font = "bold 12px monospace";
      ctx.strokeStyle = "#08120e";
      ctx.lineWidth = 3;
      ctx.strokeText("G" + id, x+3, y);
      ctx.fillText("G" + id, x+3, y);
    }
  }
}
