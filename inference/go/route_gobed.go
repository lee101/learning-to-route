// Routing as embedding search with gobed (github.com/lee101/gobed).
// Index the router anchors once at startup; each route() is one static embed
// (~0.15ms) plus one ANN search. Suitable for running inside a gateway hot path.
//
// go run route_gobed.go "fix the flaky integration test" router.json
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	gobed "github.com/lee101/gobed"
)

type anchorStats struct {
	N    float64 `json:"n"`
	Pass float64 `json:"pass"`
	Cost float64 `json:"cost"`
}

type anchor struct {
	Text  string                 `json:"text"`
	Stats map[string]anchorStats `json:"stats"`
}

type routerFile struct {
	Anchors []anchor `json:"anchors"`
}

func main() {
	query, path := os.Args[1], os.Args[2]
	raw, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	var rf routerFile
	if err := json.Unmarshal(raw, &rf); err != nil {
		panic(err)
	}

	model, err := gobed.LoadModel()
	if err != nil {
		panic(err)
	}
	engine := gobed.NewSearchEngine(model)
	texts := make([]string, len(rf.Anchors))
	for i, a := range rf.Anchors {
		texts[i] = a.Text
	}
	if _, err := engine.IndexBatch(texts); err != nil {
		panic(err)
	}

	hits, err := engine.Search(query, 8)
	if err != nil {
		panic(err)
	}
	num := map[string]float64{}
	den := map[string]float64{}
	for _, h := range hits {
		for m, s := range rf.Anchors[h.ID].Stats {
			w := float64(h.Similarity) * min(s.N, 10)
			if w <= 0 {
				continue
			}
			num[m] += w * s.Pass
			den[m] += w
		}
	}
	type ranked struct {
		Model string
		Pass  float64
	}
	var out []ranked
	for m := range num {
		out = append(out, ranked{m, num[m] / den[m]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pass > out[j].Pass })
	for _, r := range out {
		fmt.Printf("%-24s expected_pass=%.3f\n", r.Model, r.Pass)
	}
}
