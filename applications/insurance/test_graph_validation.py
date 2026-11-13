import networkx as nx

from applications.insurance.graph_validation import (
    approved_connectivity,
    graph_connectivity,
)


def test_two_node_island_fails_even_with_no_degree_zero_nodes():
    graph = nx.Graph([("产品甲", "产品甲｜责任"), ("产品甲｜流程", "产品甲｜材料")])
    report = graph_connectivity(graph, ["产品甲"])
    assert not report["passed"]
    assert report["isolated_nodes"] == []
    assert report["unreachable_by_product"]["产品甲"] == [
        "产品甲｜材料",
        "产品甲｜流程",
    ]
    graph.add_edge("产品甲", "产品甲｜流程")
    assert graph_connectivity(graph, ["产品甲"])["passed"]


def test_degree_zero_missing_root_and_cross_product_detour_fail():
    graph = nx.Graph([("产品甲", "产品乙"), ("产品乙", "产品甲｜材料")])
    assert not graph_connectivity(graph, ["产品甲", "产品乙"])["passed"]
    graph.add_node("产品乙｜孤点")
    assert graph_connectivity(graph)["isolated_nodes"] == ["产品乙｜孤点"]
    assert not graph_connectivity(nx.Graph([("产品丙｜甲", "产品丙｜乙")]))["passed"]


def test_independent_valid_products_need_not_share_a_global_component():
    graph = nx.Graph([("产品甲", "产品甲｜责任"), ("产品乙", "产品乙｜责任")])
    assert graph_connectivity(graph)["passed"]


def test_preflight_merges_chunks_before_testing_connectivity():
    def row(names, edges):
        return {
            "native": {
                "entities": [{"name": n} for n in names],
                "relationships": [{"source": a, "target": b} for a, b in edges],
            }
        }

    first = row(
        ["产品（甲）", "产品（甲）｜责任"], [("产品（甲）", "产品（甲）｜责任")]
    )
    second = row(
        ["产品（甲）｜责任", "产品（甲）｜材料"],
        [("产品（甲）｜责任", "产品（甲）｜材料")],
    )
    assert approved_connectivity({"a": first, "b": second}, ["产品（甲）"])["passed"]
    assert not approved_connectivity({"b": second}, ["产品（甲）"])["passed"]
