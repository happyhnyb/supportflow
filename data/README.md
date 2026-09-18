# Dataset provenance

`support_tickets.csv` is a deterministic, balanced 600-row subset of Bitext's
public **Retail (eCommerce) LLM Chatbot Training Dataset**:

- Dataset page: <https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset>
- Pinned source revision: `12dd624ddcd3057382b2faad661bcda1fa869491`
- Source CSV SHA-256: `13a988266fed4e2b2c1ff947a89ef220ce09b5b13ac83c4a1496c0d7b81e8127`
- Original licence: CDLA-Sharing-1.0

The source provides 44,884 English e-commerce instructions across 46 intents.
For the proof of concept, `scripts/prepare_dataset.py` selects 100 rows from
each of six relevant source intents using random seed 42, then maps them to the
six labels used by SupportFlow:

| Source intent | SupportFlow label |
| --- | --- |
| `delivery_issue` | `delivery_delay` |
| `request_refund` | `refund_request` |
| `damaged_delivery` | `damaged_item` |
| `payment_issue` | `payment_problem` |
| `recover_password` | `account_access` |
| `change_order` | `order_change` |

Each row retains `source_intent` and `source_row` so it can be traced back to
the pinned public CSV. Rebuild the subset with:

```bash
python3 scripts/prepare_dataset.py
```

The subset is redistributed under the source dataset's CDLA-Sharing-1.0 terms.
It contains generated customer-support utterances, not real customer records.
