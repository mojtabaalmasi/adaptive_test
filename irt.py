# irt.py
# -*- coding: utf-8 -*-
"""
ماژول IRT (مدل سه‌پارامتری 3PL) با پیاده‌سازی‌های پایدار:
- احتمال 3PL با کلیپ و جلوگیری از overflow
- گرادیان صحیح لگ‌لایکلیهود 3PL
- تخمین θ با نیوتن–رافسون (پایدارتر از گام ثابت)
- تابع اطلاعات آیتم/آزمون (فرمول صحیح 3PL)
- انتخاب سؤال بر مبنای بیشترین اطلاعات با گارد
- قانون توقف پیشنهادی: SE(theta) ~ 1/sqrt(I_test)
- ترسیم ICC و اطلاعات آزمون
- ذخیره‌ی نتایج در Excel/Word
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document

# ثوابت عددی
EPS = 1e-9
THETA_MIN, THETA_MAX = -4.0, 4.0


# ------------------------ توابع پایه ------------------------

def _sigmoid(x: float) -> float:
    """لاجیستیک با کلیپ برای جلوگیری از overflow."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def three_pl_probability(theta: float, a: float, b: float, c: float) -> float:
    """
    احتمال پاسخ صحیح در مدل 3PL:
    P(X=1|θ) = c + (1 - c) * logistic(a(θ - b))
    با کلیپ p برای جلوگیری از log(0) و تقسیم‌های خطرناک.
    """
    a = max(float(a), EPS)
    c = float(np.clip(c, 0.0, 0.999))
    p = c + (1.0 - c) * _sigmoid(a * (theta - b))
    return float(np.clip(p, c + EPS, 1.0 - EPS))


# ------------------------ درستنمایی و گرادیان ------------------------

def log_likelihood(theta: float, responses, item_params) -> float:
    """
    لگ‌لایکلیهود پاسخ‌ها در θ (لیست responses شامل 0/1).
    """
    ll = 0.0
    for r, (a, b, c) in zip(responses, item_params):
        p = three_pl_probability(theta, a, b, c)
        ll += r * np.log(p) + (1 - r) * np.log(1.0 - p)
    return float(ll)


def _grad_loglik_theta(theta: float, responses, item_params) -> float:
    """
    گرادیان صحیح لگ‌لایکلیهود نسبت به θ برای 3PL:
    sum_i a_i * (x_i - p_i) * (p_i - c_i) / ((1 - c_i) * p_i)
    """
    g = 0.0
    for x, (a, b, c) in zip(responses, item_params):
        p = three_pl_probability(theta, a, b, c)
        g += a * (x - p) * (p - c) / ((1.0 - c) * p + EPS)
    return float(g)


# ------------------------ اطلاعات آیتم/آزمون و SE ------------------------

def item_information(theta: float, a: float, b: float, c: float) -> float:
    """
    اطلاعات آیتم برای 3PL:
    I_i(θ) = a^2 * ((p - c)^2 / (1 - c)^2) * (q / p)
    """
    p = three_pl_probability(theta, a, b, c)
    q = 1.0 - p
    denom = (1.0 - c) ** 2 + EPS
    info = (a ** 2) * ((p - c) ** 2) / denom * (q / p)
    return float(info) if np.isfinite(info) and info > 0 else 0.0


def test_information(theta: float, item_params) -> float:
    """اطلاعات کل آزمون در θ."""
    return float(sum(item_information(theta, a, b, c) for (a, b, c) in item_params))


def theta_se(theta: float, item_params) -> float:
    """
    خطای استاندارد تخمین θ:
    SE(θ) ≈ 1 / sqrt(I_test(θ))
    """
    I = test_information(theta, item_params)
    return float(1.0 / np.sqrt(max(I, EPS)))


# ------------------------ تخمین θ ------------------------

def estimate_theta_mle(responses, item_params, lr: float = 0.01,
                       max_iter: int = 50, tol: float = 1e-4) -> float:
    """
    تخمین MLE با نیوتن–رافسون مبتنی بر اطلاعات فیشر.
    پارامتر lr فقط برای سازگاری امضا حفظ شده و استفاده نمی‌شود.
    """
    theta = 0.0
    for _ in range(max_iter):
        g = _grad_loglik_theta(theta, responses, item_params)
        I = test_information(theta, item_params) + EPS
        step = g / I

        # نرم‌سازی گام‌های بزرگ/ناپایدار
        if not np.isfinite(step) or abs(step) > 1.0:
            step = 0.25 * np.tanh(step)

        theta_new = float(np.clip(theta + step, THETA_MIN, THETA_MAX))
        if abs(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta


def estimate_theta_map(responses, item_params, theta0: float = 0.0,
                       prior_mean: float = 0.0, prior_var: float = 1.0,
                       max_iter: int = 50, tol: float = 1e-4) -> float:
    """
    تخمین MAP با prior نرمال N(prior_mean, prior_var).
    """
    theta = float(np.clip(theta0, THETA_MIN, THETA_MAX))
    inv_var = 1.0 / max(prior_var, EPS)

    for _ in range(max_iter):
        g_like = _grad_loglik_theta(theta, responses, item_params)
        g_prior = -(theta - prior_mean) * inv_var
        g = g_like + g_prior

        I = test_information(theta, item_params) + inv_var + EPS
        step = g / I
        if not np.isfinite(step) or abs(step) > 1.0:
            step = 0.25 * np.tanh(step)

        theta_new = float(np.clip(theta + step, THETA_MIN, THETA_MAX))
        if abs(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta


def estimate_theta_eap(responses, item_params, nodes: int = 41,
                       prior_mean: float = 0.0, prior_sd: float = 1.0) -> float:
    """
    تخمین EAP با گوس-هرمیت؛ پایدار برای تعداد پاسخ کم.
    """
    from numpy.polynomial.hermite import hermgauss

    z, w = hermgauss(nodes)
    thetas = np.sqrt(2.0) * prior_sd * z + prior_mean
    weights = w / np.sqrt(np.pi)

    def person_ll(th):
        s = 0.0
        for x, (a, b, c) in zip(responses, item_params):
            p = three_pl_probability(th, a, b, c)
            s += np.log(p if x else (1.0 - p))
        return s

    logliks = np.array([person_ll(th) for th in thetas])
    m = np.max(logliks)
    post = np.exp(logliks - m) * weights
    post /= (np.sum(post) + EPS)

    eap = float(np.sum(thetas * post))
    return float(np.clip(eap, THETA_MIN, THETA_MAX))


# ------------------------ انتخاب سؤال ------------------------

def select_next_question(theta: float, all_item_params, answered_indices):
    """
    انتخاب آیتم با بیشترین اطلاعات در θ.
    اگر همه پاسخ داده شده باشند، None برمی‌گرداند.
    """
    best_idx, best_info = None, -1.0
    for i, (a, b, c) in enumerate(all_item_params):
        if i in answered_indices:
            continue
        info = item_information(theta, a, b, c)
        if info > best_info:
            best_info, best_idx = info, i
    return best_idx


# ------------------------ ترسیم نمودارها ------------------------

def plot_icc(item_params, save_path='static/icc.png'):
    """
    ترسیم منحنی‌های ICC برای آیتم‌های داده‌شده.
    """
    theta_range = np.linspace(THETA_MIN, THETA_MAX, 200)
    plt.figure(figsize=(10, 6))
    for i, (a, b, c) in enumerate(item_params):
        probs = [three_pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, probs, label=f"سوال {i+1}")
    plt.xlabel('θ (توانایی)')
    plt.ylabel('احتمال پاسخ صحیح')
    plt.title('تابع مشخصه سوالات (ICC)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_item_information(item_params, save_path='static/item_info.png'):
    """
    ترسیم اطلاعات کل آزمون در بازه‌ی θ.
    """
    theta_range = np.linspace(THETA_MIN, THETA_MAX, 200)
    total_info = np.zeros_like(theta_range, dtype=float)
    for a, b, c in item_params:
        info = np.array([item_information(t, a, b, c) for t in theta_range], dtype=float)
        total_info += info
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, total_info)
    plt.xlabel('θ (توانایی)')
    plt.ylabel('اطلاعات آزمون')
    plt.title('تابع اطلاعات کل آزمون')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ------------------------ خروجی گرفتن ------------------------

def save_results_to_excel(filepath: str, responses, item_params, theta: float):
    """
    ذخیره‌ی نتایج در Excel. فرض: item_params مربوط به آیتم‌های پاسخ‌داده‌شده است.
    """
    df = pd.DataFrame({
        'سوال': [f"سوال {i+1}" for i in range(len(responses))],
        'پاسخ (0/1)': responses,
        'a': [a for a, _, _ in item_params],
        'b': [b for _, b, _ in item_params],
        'c': [c for _, _, c in item_params],
    })
    df.loc[len(df.index)] = ['θ (توانایی)', theta, '', '', '']
    df.to_excel(filepath, index=False)


def save_results_to_word(filepath: str, responses, item_params, theta: float):
    """
    ذخیره‌ی نتایج در Word (docx). فرض: item_params مربوط به آیتم‌های پاسخ‌داده‌شده است.
    """
    doc = Document()
    doc.add_heading('نتایج آزمون انطباقی (3PL)', level=1)
    doc.add_paragraph(f'مقدار تخمینی θ: {theta:.3f}')

    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    hdr[0].text = 'سوال'
    hdr[1].text = 'پاسخ (0/1)'
    hdr[2].text = 'a'
    hdr[3].text = 'b'
    hdr[4].text = 'c'

    for i, (r, (a, b, c)) in enumerate(zip(responses, item_params)):
        row = table.add_row().cells
        row[0].text = f"سوال {i+1}"
        row[1].text = str(int(r))
        row[2].text = f"{a:.3f}"
        row[3].text = f"{b:.3f}"
        row[4].text = f"{c:.3f}"

    doc.save(filepath)
