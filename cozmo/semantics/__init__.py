"""The non-geometric half of the output contract.

Tier-agnostic by construction: every entry point here takes a metric point
cloud, RGB frames and fitted surfaces, and knows nothing about how they were
obtained. A photo-tier reconstruction that produces the same three inputs gets
the same damage regions, flags and scope out.
"""

from .rules import ConcealedRule, RuleEngine, load_rules
from .scope import ScopeCatalogue, ScopeLine, load_catalogue, price_regions

__all__ = [
    "ConcealedRule", "RuleEngine", "ScopeCatalogue", "ScopeLine",
    "load_catalogue", "load_rules", "price_regions",
]
