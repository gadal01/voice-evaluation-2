"""Sample "service desk" MCP server: catalog, pricing, scheduling and request creation.

Mounted into the FastAPI app at /mcp (streamable HTTP) by main.py, and also runnable
on its own over stdio:   python mcp_server.py
All data is in memory and made up; replace the helpers with calls to your real systems.
"""
import itertools
from datetime import date, timedelta
from typing import Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "service-desk",
    instructions="Look up serviceable items, quote servicing costs and create service requests.",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",  # main.py mounts this app under /mcp
)

# --------------------------------------------------------------- sample data --
# base_cost = diagnostic/inspection fee in USD; repair & full service are multiples of it.
CATALOG = {
    "AC-100": {"name": "Split Air Conditioner", "category": "appliance", "base_cost": 49, "hours": 1.5},
    "RF-200": {"name": "Refrigerator", "category": "appliance", "base_cost": 59, "hours": 2.0},
    "WM-300": {"name": "Washing Machine", "category": "appliance", "base_cost": 45, "hours": 1.5},
    "WH-400": {"name": "Water Heater", "category": "plumbing", "base_cost": 69, "hours": 2.5},
    "LP-500": {"name": "Laptop", "category": "electronics", "base_cost": 39, "hours": 1.0},
    "TV-600": {"name": "Television", "category": "electronics", "base_cost": 35, "hours": 1.0},
}
SERVICE_MULTIPLIER = {"inspection": 1.0, "repair": 2.5, "full_service": 4.0}
URGENCY_MULTIPLIER = {"standard": 1.0, "express": 1.5}
PARTS_ESTIMATE = {"inspection": 0, "repair": 40, "full_service": 25}
TAX_RATE = 0.08

REQUESTS: dict[str, dict] = {}
_ids = itertools.count(1001)

ServiceType = Literal["inspection", "repair", "full_service"]
Urgency = Literal["standard", "express"]


def _quote(item_id: str, service_type: str, urgency: str) -> dict:
    item = CATALOG[item_id]
    labor = round(item["base_cost"] * SERVICE_MULTIPLIER[service_type] * URGENCY_MULTIPLIER[urgency], 2)
    parts = PARTS_ESTIMATE[service_type]
    subtotal = labor + parts
    tax = round(subtotal * TAX_RATE, 2)
    return {
        "item_id": item_id, "item": item["name"], "service_type": service_type, "urgency": urgency,
        "labor": labor, "parts_estimate": parts, "tax": tax, "total_usd": round(subtotal + tax, 2),
        "estimated_hours": item["hours"],
    }


_SPOKEN_DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                  "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _normalize_phone(phone: str) -> str:
    """Speech models often pass numbers as words ("five five five oh one..."); store digits only."""
    words = phone.lower().replace("-", " ").split()
    return "".join(_SPOKEN_DIGITS.get(w.strip(".,"), w) for w in words if w) if any(
        w.strip(".,") in _SPOKEN_DIGITS for w in words) else phone


def _slots(urgency: str) -> list[str]:
    start = date.today() + timedelta(days=1 if urgency == "express" else 3)
    out = []
    d = start
    while len(out) < 3:
        if d.weekday() < 6:  # Mon-Sat
            out.append(f"{d.isoformat()} 10:00-12:00" if len(out) % 2 == 0 else f"{d.isoformat()} 14:00-16:00")
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------- tools --
@mcp.tool()
def list_serviceable_items(category: str | None = None) -> list[dict]:
    """List items we can service, with id, name, category and inspection fee.
    Call it with NO category unless the caller explicitly asked for a category. Categories are
    appliance (air conditioner, refrigerator, washing machine), plumbing (water heater) and
    electronics (laptop, television). Never guess a category: if an item is not in a filtered
    list, call again without a category before telling the caller we can't service it."""
    return [
        {"item_id": k, "name": v["name"], "category": v["category"], "inspection_fee_usd": v["base_cost"]}
        for k, v in CATALOG.items()
        if category is None or v["category"] == category.lower()
    ]


@mcp.tool()
def get_servicing_cost(item_id: str, service_type: ServiceType = "inspection", urgency: Urgency = "standard") -> dict:
    """Quote the servicing cost (labor, estimated parts, tax, total in USD) for an item.
    service_type: inspection | repair | full_service. urgency: standard | express (+50%)."""
    item_id = item_id.upper()
    if item_id not in CATALOG:
        return {"error": f"Unknown item_id {item_id}. Call list_serviceable_items first."}
    return _quote(item_id, service_type, urgency)


@mcp.tool()
def get_available_slots(urgency: Urgency = "standard") -> dict:
    """Return the next available technician appointment slots.
    Express service has earlier slots than standard."""
    return {"urgency": urgency, "slots": _slots(urgency)}


@mcp.tool()
def create_service_request(
    customer_name: str,
    phone: str,
    item_id: str,
    issue_description: str,
    service_type: ServiceType = "inspection",
    urgency: Urgency = "standard",
    preferred_slot: str | None = None,
    brand: str | None = None,
) -> dict:
    """Create a service request. Only call after the customer has confirmed the details and quoted cost.
    brand is the make of the item (for example Samsung or LG) if the caller said it; the symptom goes in issue_description.
    preferred_slot must be one of get_available_slots; defaults to the earliest slot."""
    item_id = item_id.upper()
    if item_id not in CATALOG:
        return {"error": f"Unknown item_id {item_id}. Call list_serviceable_items first."}
    slots = _slots(urgency)
    if preferred_slot and preferred_slot not in slots:
        return {"error": "preferred_slot not available", "available_slots": slots}
    request_id = f"SR-{next(_ids)}"
    REQUESTS[request_id] = {
        "request_id": request_id, "status": "scheduled", "customer_name": customer_name,
        "phone": "".join(ch for ch in _normalize_phone(phone) if ch.isdigit()) or phone,
        "issue_description": issue_description, "brand": brand, "slot": preferred_slot or slots[0],
        "quote": _quote(item_id, service_type, urgency),
    }
    return REQUESTS[request_id]


@mcp.tool()
def get_service_request(request_id: str) -> dict:
    """Look up the status and details of an existing service request, e.g. SR-1001."""
    return REQUESTS.get(request_id.upper()) or {"error": f"No request {request_id}"}


@mcp.tool()
def cancel_service_request(request_id: str) -> dict:
    """Cancel a scheduled service request."""
    req = REQUESTS.get(request_id.upper())
    if not req:
        return {"error": f"No request {request_id}"}
    req["status"] = "cancelled"
    return req


if __name__ == "__main__":
    mcp.run(transport="stdio")
