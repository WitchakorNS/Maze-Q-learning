"""Readable phase-5 report, generated from measured results rather than claims."""


def write_bonus_report(run_dir, result, training):
    cfg = result["config"]
    methods = cfg["methods"]
    metrics = {row["method"]: row for row in result["comparison"]}
    count = len(cfg["seeds"])

    def fmt(value):
        if value is None:
            return "ไม่พบ"
        return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"

    metric_rows = []
    for label, key in (
        ("ก้าวฝึกต่อ seed (รวม warm-up/ชนกำแพง)", "training_actions_per_seed"),
        ("โมเดลที่สำเร็จสุดท้าย", "successes"),
        ("Success rate สุดท้าย (%)", "success_rate_percent"),
        ("ก้าวเฉลี่ยทดสอบสุดท้าย (รวมล้มเหลว)", "mean_steps_all"),
        ("รางวัลเกมเฉลี่ยทดสอบสุดท้าย", "mean_env_reward"),
        ("ก้าวฝึกเฉลี่ยที่พบ greedy success ครั้งแรก*", "mean_first_observed_success_step"),
        ("โมเดลที่เคยพบ greedy success", "models_with_observed_success"),
        ("ก้าวเริ่มต้นเฉลี่ยของ success 3 จุดติดกัน*", "mean_three_point_success_start_step"),
        ("โมเดลที่มี success 3 จุดติดกัน", "models_with_three_point_success"),
    ):
        is_count = key in ("successes", "models_with_observed_success", "models_with_three_point_success")
        values = [f"{metrics[m][key]}/{count}" if is_count else fmt(metrics[m][key]) for m in methods]
        metric_rows.append(f"| {label} | " + " | ".join(values) + " |")

    curr, bonus = metrics["curriculum"], metrics["curriculum_bonus"]
    if curr["models_with_observed_success"] == bonus["models_with_observed_success"] == count:
        delta = bonus["mean_first_observed_success_step"] - curr["mean_first_observed_success_step"]
        if delta < 0:
            finding = f"การเพิ่มโบนัสบน Curriculum พบ greedy success ครั้งแรกเร็วขึ้นเฉลี่ย {-delta:,.0f} ก้าวฝึก"
        elif delta > 0:
            finding = f"การเพิ่มโบนัสบน Curriculum พบ greedy success ครั้งแรกช้าลงเฉลี่ย {delta:,.0f} ก้าวฝึก"
        else:
            finding = "การเพิ่มโบนัสบน Curriculum ให้ค่าเฉลี่ยจุดที่พบ greedy success ครั้งแรกเท่ากัน"
    else:
        finding = "บาง seeds ยังไม่พบ success ภายในงบ ต้องอ่านจำนวนโมเดลที่พบเกณฑ์ร่วมกับค่าเฉลี่ย ไม่ตัดรอบล้มเหลวทิ้งเพื่อสรุปความเร็ว"
    seed_rows = "\n".join(
        f"| {r['method']} | {r['seed']} | {fmt(r['first_observed_success_step'])} | "
        f"{fmt(r['three_point_success_start_step'])} | {'สำเร็จ' if r['success'] else 'หมดก้าว'} | "
        f"{r['steps']} | {r['env_reward']} | {r['wall_hits']} |" for r in result["per_seed"])
    bonus_rows = []
    for method in methods:
        rows = [r for r in training if r["method"] == method]
        bonus_rows.append(f"| {method} | {sum(r['steps'] for r in rows):,} | "
                          f"{sum(r['env_reward'] for r in rows):,.2f} | "
                          f"{sum(r['intrinsic_reward'] for r in rows):,.2f} | "
                          f"{sum(r['learning_reward'] for r in rows):,.2f} |")
    figure = (f"![Learning curve]({result['learning_curve']})" if result["learning_curve"]
              else "ไม่ได้สร้างกราฟ ใช้ logs/evaluation.csv วาดได้")
    command = (
        f"python exploration.py --budget {cfg['training_actions_per_seed']} "
        f"--seeds {' '.join(map(str, cfg['seeds']))} --bonus-beta {cfg['bonus_beta']} "
        f"--max-steps {result['environment']['max_steps']} --eval-every {cfg['evaluation_every_actions']} "
        f"--alpha {cfg['alpha']} --gamma {cfg['gamma']} --epsilon-start {cfg['epsilon_start']} "
        f"--epsilon-min {cfg['epsilon_min']} --epsilon-decay {cfg['epsilon_decay_per_action']} "
        f"--easy-radius {cfg['easy_radius']} --medium-radius {cfg['medium_radius']} "
        f"--window {cfg['promotion_window_complete_episodes']} "
        f"--threshold {cfg['promotion_training_success_threshold']} --stage-cap {cfg['warmup_stage_action_cap']}"
    )
    report = f"""# Exploration Bonus — ผลระยะที่ 5

รันเมื่อ {result['created_at']} (UTC+07:00) ใช้หน้านี้กับหน้าจอเดโมนำเสนอได้เลย

## ผลเปรียบเทียบ

ฝึกใหม่ทุกวิธีจาก Q=0 ด้วย seeds {cfg['seeds']} วิธีละ {count} โมเดล งบ {cfg['training_actions_per_seed']:,} action ต่อโมเดล

| ตัวชี้วัด | Q ปกติ | Curriculum | Curriculum + Bonus |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

จำนวนโมเดลในตารางอ่านเทียบกับ {count} โมเดลต่อวิธี

{finding} ในแผนที่และการตั้งค่าชุดนี้ ผลสุดท้ายอ่านจาก success/ก้าว/รางวัลเกมในตาราง ไม่ใช่จากรางวัลรวมโบนัสที่ใช้สอน

*เป็นจุดที่สังเกตจากการประเมินทุก {cfg['evaluation_every_actions']:,} action และจุด snapshot ที่กำหนดไว้ล่วงหน้า ไม่ใช่ก้าวที่เรียนรู้ได้ครั้งแรกอย่างแน่นอน ค่าเฉลี่ยเวลาเฉพาะโมเดลที่พบเกณฑ์ต้องอ่านคู่จำนวนโมเดล และ success 3 จุดติดกันไม่ได้รับประกันความสำเร็จทุกครั้งหลังจากนั้น

{figure}

ซ้ายแสดงเต็มงบ ขวาขยาย 20% แรก ประเมินจาก S จริงด้วย ε=0 ไม่มีโบนัส ไม่มี update และไม่ใช้ผลประเมินหยุดฝึก เลื่อนระดับ หรือเลือก checkpoint ที่ดีที่สุด

## โบนัสทำงานอย่างไร

ระหว่างฝึกใช้ `learning_reward = env_reward + intrinsic_reward`

เมื่อเดินเข้าช่องอื่นที่ยังไม่ใช่เป้าหมาย ให้ `intrinsic_reward = beta / sqrt(N(next_state) + 1)` โดย N คือจำนวนครั้งที่เข้าช่องนั้น **ก่อน** action นี้ แล้วจึงเพิ่มตัวนับ ตั้ง beta={cfg['bonus_beta']} ไว้ก่อนทดลอง ไม่ปรับย้อนหลังจากผล

ตัวอย่าง beta=0.5: เข้าช่องใหม่ได้โบนัส 0.5, ครั้งที่ 2 ประมาณ 0.354, ครั้งที่ 3 ประมาณ 0.289 รางวัลการเดินปกติ -1 จึงกลายเป็น -0.5 ในครั้งแรก ไม่ได้ทำให้การเดินวนมีรางวัลบวกทันที

- ชนกำแพง: โบนัส 0 และไม่นับเป็นการเข้าช่องใหม่
- ถึง G: โบนัส 0 แต่เพิ่มจำนวนการเข้าช่อง G; รางวัลเกมยังเป็น +100
- การ reset: นับจุดเริ่มเป็นหนึ่ง visit แต่ไม่ให้โบนัสและไม่ใช้ action ฝึก
- ตัวนับเริ่มใหม่ต่อ training seed และคงอยู่ข้าม episode/ระดับ Curriculum ไม่ reset เมื่อเลื่อนระดับ
- การประเมินและเดโมไม่เรียกตัวนับ ไม่เพิ่ม visit และไม่รับโบนัส แม้โหลดโมเดลจากวิธี Bonus
- โมเดลมี snapshot ตัวนับและโบนัสสะสมใน metadata เพื่อดูหลักฐาน ไม่ได้รองรับการ resume ฝึกแบบ RNG ตรงทุกก้าว

นี่เป็น **count-based exploration bonus** ไม่ใช่ curiosity ที่ใช้ neural network ทำนาย next state แบบคลิปต้นฉบับ ตัวนับใช้เฉพาะตำแหน่งจาก transition และสถานะจบรอบ ไม่อ่านแผนที่หรือเส้นทาง

## แยกรางวัลให้ถูกตอนรายงาน

| วิธี | Training actions รวมทุก seed | รางวัลเกมสะสมระหว่างฝึก | โบนัสสะสมระหว่างฝึก | รางวัลที่ใช้ update สะสม |
|---|---:|---:|---:|---:|
{chr(10).join(bonus_rows)}

ตารางนี้ตรวจสอบบัญชีรางวัลระหว่างฝึก ไม่ใช้จัดอันดับความสามารถ: Curriculum มีจุดเริ่มใกล้ G ระหว่างฝึก และวิธี Bonus เพิ่มรางวัลภายใน จึงเปรียบเทียบคะแนนฝึกดิบตรง ๆ ไม่ได้ ตัวชี้วัดหลักอยู่ตารางแรกซึ่งประเมินด้วยรางวัลเกมจาก S เดียวกัน

## ผลแยก seed

| วิธี | Seed | พบ success ครั้งแรก* | เริ่ม success 3 จุดติดกัน* | สุดท้าย | ก้าว | รางวัลเกม | ชนกำแพง |
|---|---:|---:|---:|---|---:|---:|---:|
{seed_rows}

## เงื่อนไขร่วมและขอบเขต

ทุกวิธีใช้แผนที่/เป้าหมาย/กติกาเดิม alpha={cfg['alpha']}, gamma={cfg['gamma']}, epsilon เริ่ม {cfg['epsilon_start']} คูณ {cfg['epsilon_decay_per_action']} ต่อ action ต่ำสุด {cfg['epsilon_min']} ไม่มี action masking งบรวมการชนกำแพงและ warm-up เสมอ episode ที่ถูกตัดเพราะงบหมดบันทึกแยกและไม่ใช้เลื่อนระดับ

Curriculum ทั้งสองแบบใช้ near ระยะ 1–{cfg['easy_radius']}, middle {cfg['easy_radius']+1}–{cfg['medium_radius']} แล้ว S จริง เลื่อนระดับจาก success ≥{cfg['promotion_training_success_threshold']:.0%} ใน {cfg['promotion_window_complete_episodes']} episode สมบูรณ์ล่าสุด หรือ stage cap {cfg['warmup_stage_action_cap']:,} action (ตรวจที่ขอบ episode) ไม่ reset epsilon ผู้สอนใช้ BFS จัดจุดเริ่ม แต่ Agent ไม่ได้รับระยะ/route

การเทียบ Curriculum + Bonus กับ Curriculum ศึกษาผลเพิ่มโบนัสเมื่อมี Curriculum อยู่แล้ว ยังไม่มี Q + Bonus เพียงอย่างเดียว จึงยังแยกปฏิสัมพันธ์ของสองเทคนิคทั้งหมดไม่ได้ Beta อื่นยังไม่ได้ทดสอบ

ประเมินหนึ่ง greedy rollout ต่อ training seed เพราะนโยบายและ Environment เป็น deterministic การรันซ้ำโมเดลเดิมไม่เพิ่มตัวอย่างอิสระ ยังไม่ทดสอบแผนที่ใหม่หรือจุดเริ่มอื่น ไม่อ้างว่าเทคนิคนี้ดีกว่าเสมอหรือมีนัยสำคัญทางสถิติจาก 5 seeds

## โชว์ในคลาส

1. เปิด play.cmd ให้ Random เดินเพื่อเห็นปัญหา
2. เลือก Q-learning, Curriculum หรือ Curric + Bonus แล้วกด ให้ Q เดิน ทั้งสามดึงโมเดล seed แรกของการทดลองเดียวกัน ไม่เลือก seed ที่ดีที่สุด
3. เปิดรายงานนี้ ชี้กราฟช่วงต้นและตารางผล อธิบายว่าโบนัสเปลี่ยนการฝึกอย่างไร แต่คะแนนทดสอบยังใช้กติกาเดิม
4. ถ้าต้องการโชว์ระหว่างฝึก ใช้ โหลด Q-table เลือก snapshot ที่จำนวน action เดียวกันจาก models/ โมเดลสุดท้ายอาจเดินเหมือนกัน แม้เรียนรู้ต่างกัน

ไฟล์ที่ใช้โชว์หลักคือ summary.md และหน้าจอเกม หลักฐานละเอียดอยู่ logs/ (training, evaluation, final_evaluation, stages) และ models/ มี config/hash ใน summary.json

## ทำซ้ำ

```powershell
{command}
```

หรือดับเบิลคลิก exploration.cmd สำหรับค่าเริ่มต้น ใช้ --no-plot หากไม่มี Matplotlib รันใหม่สร้างโฟลเดอร์ใหม่ ไม่มีการทับผลเก่า
"""
    with (run_dir / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
