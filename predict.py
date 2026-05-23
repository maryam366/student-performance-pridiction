"""
predict.py — Predict outcome for a single student (Real UCI Data)
Usage: python src/predict.py
"""

import pandas as pd
import numpy as np
import joblib
import glob

def predict_student(student: dict) -> dict:
    model_files = glob.glob('models/*.pkl')
    if not model_files:
        print("❌ No model found. Run src/train_evaluate.py first.")
        return {}

    model = joblib.load(model_files[0])

    FEATURES = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                'absences', 'G1', 'G2', 'course']

    cat_maps = {
        'school':    {'GP': 0, 'MS': 1},
        'sex':       {'F': 0, 'M': 1},
        'address':   {'R': 0, 'U': 1},
        'famsize':   {'GT3': 0, 'LE3': 1},
        'Pstatus':   {'A': 0, 'T': 1},
        'Mjob':      {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
        'Fjob':      {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
        'reason':    {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
        'guardian':  {'father': 0, 'mother': 1, 'other': 2},
        'schoolsup': {'no': 0, 'yes': 1},
        'famsup':    {'no': 0, 'yes': 1},
        'paid':      {'no': 0, 'yes': 1},
        'activities':{'no': 0, 'yes': 1},
        'nursery':   {'no': 0, 'yes': 1},
        'higher':    {'no': 0, 'yes': 1},
        'internet':  {'no': 0, 'yes': 1},
        'romantic':  {'no': 0, 'yes': 1},
        'course':    {'Math': 0, 'Portuguese': 1},
    }

    encoded = student.copy()
    for col, mapping in cat_maps.items():
        if col in encoded:
            encoded[col] = mapping.get(str(encoded[col]), 0)

    X = pd.DataFrame([encoded])[FEATURES]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return {
        'prediction':       'Pass ✅' if pred == 1 else 'Fail ❌',
        'pass_probability': f'{prob[1]*100:.1f}%',
        'fail_probability': f'{prob[0]*100:.1f}%',
    }


if __name__ == '__main__':
    # Sample student
    student = {
        'school': 'GP', 'sex': 'F', 'age': 16,
        'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
        'Medu': 3, 'Fedu': 2,
        'Mjob': 'health', 'Fjob': 'other',
        'reason': 'reputation', 'guardian': 'mother',
        'traveltime': 1, 'studytime': 3, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
        'activities': 'yes', 'nursery': 'yes', 'higher': 'yes',
        'internet': 'yes', 'romantic': 'no',
        'famrel': 4, 'freetime': 3, 'goout': 2,
        'Dalc': 1, 'Walc': 1, 'health': 4,
        'absences': 2, 'G1': 13, 'G2': 14,
        'course': 'Portuguese'
    }

    print("=" * 55)
    print("  STUDENT PERFORMANCE PREDICTOR (Real UCI Data)")
    print("=" * 55)
    print("\n📋 Student Profile:")
    for k, v in student.items():
        print(f"   {k:<15}: {v}")

    result = predict_student(student)
    print(f"\n🎯 Prediction      : {result.get('prediction')}")
    print(f"   Pass Probability : {result.get('pass_probability')}")
    print(f"   Fail Probability : {result.get('fail_probability')}")
    print("=" * 55)
