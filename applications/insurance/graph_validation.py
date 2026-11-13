"""Insurance graph acceptance: check complete graphs, not a UI's bounded view."""

import networkx as nx

from lightrag.utils import normalize_entity_name


def graph_connectivity(graph, products=()):
    """Require every scoped node to reach its own product within that scope.

    Separate valid products need not share an insurer or a global component.
    Edges through another product cannot disguise missing local attribution.
    No edge is inferred or created by this check.
    """
    graph = graph.to_undirected()
    roots = {normalize_entity_name(p) for p in products}
    roots.update(name.split("｜", 1)[0] for name in graph if "｜" in name)
    components = sorted(
        (sorted(c) for c in nx.connected_components(graph)),
        key=lambda c: (-len(c), c),
    )
    isolated = sorted(nx.isolates(graph))
    unreachable = {}
    missing = []
    for root in sorted(roots):
        members = {n for n in graph if n == root or n.startswith(root + "｜")}
        if root not in graph:
            missing.append(root)
            unreachable[root] = sorted(members)
            continue
        reachable = nx.node_connected_component(graph.subgraph(members), root)
        if members - reachable:
            unreachable[root] = sorted(members - reachable)
    rootless = [c for c in components if not roots.intersection(c)]
    return {
        "passed": bool(graph) and not (isolated or missing or unreachable or rootless),
        "component_count": len(components),
        "component_sizes": [len(c) for c in components],
        "isolated_nodes": isolated,
        "rootless_components": rootless,
        "missing_product_roots": missing,
        "unreachable_by_product": unreachable,
    }


def approved_connectivity(approved, products):
    graph = nx.Graph()
    for row in approved.values():
        native = row["native"]
        graph.add_nodes_from(
            normalize_entity_name(e["name"]) for e in native["entities"]
        )
        graph.add_edges_from(
            (normalize_entity_name(e["source"]), normalize_entity_name(e["target"]))
            for e in native["relationships"]
        )
    return graph_connectivity(graph, products)
