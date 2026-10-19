"""
FraudWatch Case Management & Scoring Demonstration Script
"""
import urllib.request
import json
import time

def run_case_demo():
    tx_id = f"TX-DEMO-LIVE-{int(time.time())}"
    payload = {
        "transaction_id": tx_id,
        "step": 720,
        "type": "TRANSFER",
        "amount": 350000.0,
        "nameOrig": "C888111222",
        "oldbalanceOrg": 350000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C999333444",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }

    # 1. Score pre-auth transaction
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict/preauth",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        score_data = json.loads(res.read().decode())
    print(f"1. Scored Transaction {tx_id}:")
    print(f"   Probability: {score_data['fraud_probability']:.4f}")
    print(f"   Risk Level:  {score_data['risk_level']}")
    print(f"   Action:      {score_data['recommended_action']}")

    # 2. Query case from database via API
    with urllib.request.urlopen(f"http://127.0.0.1:8000/platform/cases/{tx_id}") as req:
        case_data = json.loads(req.read().decode())
    print(f"\n2. Case Automatically Persisted:")
    print(f"   Transaction ID: {case_data['transaction_id']}")
    print(f"   Initial Status: {case_data['case_status']}")

    # 3. Submit analyst disposition update
    review_payload = {
        "status": "INVESTIGATING",
        "disposition": "FLAGGED_SUSPICIOUS",
        "analyst_id": "analyst_lead",
        "notes": "Account drain detected; investigating related recipient accounts."
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:8000/platform/cases/{tx_id}/review",
        data=json.dumps(review_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        rev_res = json.loads(res.read().decode())
    print(f"\n3. Analyst Review 1 Submitted:")
    print(f"   New Status:  {rev_res['new_status']}")
    print(f"   Disposition: {rev_res['disposition']}")

    # 4. Submit final fraud confirmation
    review_payload_2 = {
        "status": "CONFIRMED_FRAUD",
        "disposition": "BLOCK_AND_REPORT_AML",
        "analyst_id": "analyst_lead",
        "notes": "Confirmed synthetic mule account liquidation; freezing funds."
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:8000/platform/cases/{tx_id}/review",
        data=json.dumps(review_payload_2).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as res:
        rev_res_2 = json.loads(res.read().decode())
    print(f"\n4. Analyst Review 2 Submitted:")
    print(f"   New Status:  {rev_res_2['new_status']}")
    print(f"   Disposition: {rev_res_2['disposition']}")

    # 5. Verify full audit trail
    with urllib.request.urlopen(f"http://127.0.0.1:8000/platform/cases/{tx_id}") as req:
        final_case = json.loads(req.read().decode())
    print(f"\n5. Final Verified Case Record:")
    print(f"   Case Status: {final_case['case_status']}")
    print(f"   Audit History ({len(final_case['audit_history'])} entries):")
    for entry in final_case["audit_history"]:
        print(f"     - [Disposition: {entry['disposition']}] {entry['previous_status']} -> {entry['new_status']} by {entry['analyst_id']}: '{entry['notes']}'")

if __name__ == "__main__":
    run_case_demo()
