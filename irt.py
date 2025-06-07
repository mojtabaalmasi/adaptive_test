import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document

# تابع احتمال پاسخ درست با مدل سه‌پارامتری (3PL)
def three_pl_probability(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

# تابع درستنمایی کل
def log_likelihood(theta, responses, item_params):
    prob = [three_pl_probability(theta, a, b, c) for a, b, c in item_params]
    likelihood = [
        r * np.log(p + 1e-9) + (1 - r) * np.log(1 - p + 1e-9)
        for r, p in zip(responses, prob)
    ]
    return np.sum(likelihood)

# تخمین θ با الگوریتم MLE (گرادیان صعودی ساده)
def estimate_theta_mle(responses, item_params, lr=0.01, max_iter=500, tol=1e-5):
    theta = 0.0  # مقدار اولیه
    for _ in range(max_iter):
        grad = 0
        for i, (a, b, c) in enumerate(item_params):
            p = three_pl_probability(theta, a, b, c)
            q = 1 - p
            dL = a * (responses[i] - p) * (1 - c) / (p * q + 1e-9)
            grad += dL
        theta_new = theta + lr * grad
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta

# تابع اطلاعات آیتم در θ
def item_information(theta, a, b, c):
    p = three_pl_probability(theta, a, b, c)
    q = 1 - p
    return (a ** 2) * ((p - c) ** 2) / ((1 - c) ** 2 * p * q + 1e-9)

# انتخاب سؤال بعدی با بیشترین اطلاعات در θ فعلی
def select_next_question(theta, all_item_params, answered_indices):
    infos = []
    for i, (a, b, c) in enumerate(all_item_params):
        if i in answered_indices:
            infos.append(-np.inf)  # سوالی که پاسخ داده شده را حذف کن
        else:
            info = item_information(theta, a, b, c)
            infos.append(info)
    next_q_index = np.argmax(infos)
    return next_q_index

# ترسیم نمودار تابع مشخصه سوالات (ICC)
def plot_icc(item_params, save_path='icc.png'):
    theta_range = np.linspace(-4, 4, 100)
    plt.figure(figsize=(10, 6))
    for i, (a, b, c) in enumerate(item_params):
        probs = [three_pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, probs, label=f"Q{i+1}")
    plt.xlabel('θ (توانایی)')
    plt.ylabel('P(پاسخ درست)')
    plt.title('نمودار تابع مشخصه سوالات (ICC)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ترسیم نمودار اطلاعات آزمون
def plot_item_information(item_params, save_path='item_info.png'):
    theta_range = np.linspace(-4, 4, 100)
    total_info = np.zeros_like(theta_range)
    for a, b, c in item_params:
        info = [item_information(t, a, b, c) for t in theta_range]
        total_info += info
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, total_info, color='darkblue')
    plt.xlabel('θ (توانایی)')
    plt.ylabel('اطلاعات آزمون')
    plt.title('نمودار تابع اطلاعات آزمون')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ذخیره نتایج در فایل Excel
def save_results_to_excel(filepath, responses, item_params, theta):
    df = pd.DataFrame({
        'Question': [f"Q{i+1}" for i in range(len(responses))],
        'Response': responses,
        'a': [a for a, _, _ in item_params],
        'b': [b for _, b, _ in item_params],
        'c': [c for _, _, c in item_params],
    })
    df.loc[len(df.index)] = ['θ (توانایی)', theta, '', '', '']
    df.to_excel(filepath, index=False)

# ذخیره نتایج در فایل Word
def save_results_to_word(filepath, responses, item_params, theta):
    doc = Document()
    doc.add_heading('نتایج آزمون انطباقی (3PL)', level=1)
    doc.add_paragraph(f'مقدار تخمینی θ: {theta:.3f}')
    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    hdr[0].text = 'سوال'
    hdr[1].text = 'پاسخ'
    hdr[2].text = 'a'
    hdr[3].text = 'b'
    hdr[4].text = 'c'
    for i, (r, (a, b, c)) in enumerate(zip(responses, item_params)):
        row = table.add_row().cells
        row[0].text = f"Q{i+1}"
        row[1].text = str(r)
        row[2].text = f"{a:.2f}"
        row[3].text = f"{b:.2f}"
        row[4].text = f"{c:.2f}"
    doc.save(filepath)
