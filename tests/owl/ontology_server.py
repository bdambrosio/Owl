from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rdflib
from typing import List, Any

# Define a Pydantic model for the request body
class SPARQLQuery(BaseModel):
    query: str

# Initialize the FastAPI app
app = FastAPI()

# Load the RDF graph globally so it's loaded once when the server starts
g = rdflib.Graph()
g.parse("/home/bruce/Downloads/ontology/go.owl", format="application/rdf+xml") # gene ontology - functions of genes
g.parse("/home/bruce/Downloads/ontology/cob.owl", format="application/rdf+xml")
g.parse("/home/bruce/Downloads/ontology/ro.owl", format="application/rdf+xml") # basic relation
g.parse("/home/bruce/Downloads/ontology/cl.owl", format="application/rdf+xml") # cell
#g.parse("/home/bruce/Downloads/ontology/clo.owl", format="application/rdf+xml") # cell line
g.parse("/home/bruce/Downloads/ontology/pathway.owl", format="application/rdf+xml") # pathway ontology
g.parse("/home/bruce/Downloads/ontology/ogg-merged.owl", format="application/rdf+xml") # genens and genomes
g.parse("/home/bruce/Downloads/ontology/omit.owl", format="application/rdf+xml") # microRNA targets
#g.parse("/home/bruce/Downloads/ontology/go.owl", format="application/rdf+xml")
#g.parse("/home/bruce/Downloads/ontology/go.owl", format="application/rdf+xml")
print('RDF loaded')

# Define an endpoint to accept SPARQL query and return results
@app.post("/sparql")
async def run_sparql_query(request: SPARQLQuery):
    try:
        # Execute the SPARQL query
        qres = g.query(request.query)
        
        # Prepare the results
        results = []
        for row in qres:
            # Convert each row to a dict of variable names to values
            result = {str(var): str(row[var]) for var in qres.vars}
            results.append(result)
        
        return {"results": results}
    except Exception as e:
        # If there's an error, return it
        raise HTTPException(status_code=500, detail=str(e))

