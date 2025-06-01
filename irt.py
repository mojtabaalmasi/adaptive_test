import math

def irt_1pl(theta, b):
    # مدل 1PL یا Rasch
    return 1 / (1 + math.exp(-(theta - b)))

def irt_2pl(theta, a, b):
    return 1 / (1 + math.exp(-a * (theta - b)))

def irt_3pl(theta, a, b, c):
    p = c + (1 - c) / (1 + math.exp(-a * (theta - b)))
    return p

def fisher_information(theta, a, b):
    # اطلاعات فیشر برای مدل 1PL ساده (a=1)
    p = irt_1pl(theta, b)
    q = 1 - p
    return p * q

def estimate_theta(answers, questions):
    # برآورد توانایی (theta) با روش تقریبی MLE ساده
    # برای سادگی، میانگین پارامتر b سوالاتی که درست پاسخ داده شده است را برمی‌گرداند
    correct_bs = []
    incorrect_bs = []
    for q in questions:
        qid = q['id']
        if qid in answers:
            if answers[qid] == q['correct_option']:
                correct_bs.append(q['b'])
            else:
                incorrect_bs.append(q['b'])
    if len(correct_bs) == 0:
        return 0.0
    return sum(correct_bs) / len(correct_bs)