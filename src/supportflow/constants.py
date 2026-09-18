INTENTS = (
    "delivery_delay",
    "refund_request",
    "damaged_item",
    "payment_problem",
    "account_access",
    "order_change",
)

INTENT_NAMES = {
    "delivery_delay": "Delivery delay",
    "refund_request": "Refund request",
    "damaged_item": "Damaged item",
    "payment_problem": "Payment problem",
    "account_access": "Account access",
    "order_change": "Order change",
}

RECOMMENDATIONS = {
    "delivery_delay": "Check tracking and promised delivery date; apply the late-delivery policy if eligible.",
    "refund_request": "Confirm eligibility and return status before issuing or escalating a refund.",
    "damaged_item": "Request only the minimum evidence needed and arrange a replacement or refund under policy.",
    "payment_problem": "Check payment status; never request full card or bank details in the ticket.",
    "account_access": "Use the approved identity-verification and password-reset flow.",
    "order_change": "Check fulfilment status before making a cancellation, address, or item change.",
}
