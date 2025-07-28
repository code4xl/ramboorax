import networkx as nx

async def execute_workflow(workflow):
    nodes = {node['id']: node for node in workflow['nodes']}
    edges = workflow['edges']

    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    ordered_nodes = list(nx.topological_sort(G))
    node_results = {}

    for node_id in ordered_nodes:
        node = nodes[node_id]
        node_type = node['type']

        # Fetch inputs from previous nodes
        input_data = []
        for pred in G.predecessors(node_id):
            input_data.append(node_results[pred])

        if node_type == 'customInput':
            result = node['data']['query']
        elif node_type == 'toolNode':
            result = await execute_tool_node(node, input_data)
        elif node_type == 'llm':
            result = await execute_llm_node(node, input_data)
        elif node_type == 'customOutput':
            result = input_data[0] if input_data else None

        node_results[node_id] = result

    return node_results

async def execute_tool_node(node, input_data):
    from app.services.gmail import fetch_emails, send_emails
    
    if node['data']['selectedTool'] == 'Gmail':
        if node['data']['toolActions'] == 'fetch_emails':
            return await fetch_emails(node['data']['connections']['gmail'], input_data)
        elif node['data']['toolActions'] == 'send_emails':
            return await send_emails(node['data']['connections']['gmail'], input_data)
    
    # Add other tools here as needed
    return {"error": "Unsupported tool or action"}

async def execute_llm_node(node, input_data):
    from app.services.llm import call_llm
    return await call_llm(node['data'], input_data)
