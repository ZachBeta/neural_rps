package neat

import (
	"reflect"
	"testing"
)

func TestGenomeNetworkRoundTrip(t *testing.T) {
	cfg := Config{
		PopSize:         1,
		Generations:     1,
		MutRate:         0.1,
		CxRate:          0.5,
		CompatThreshold: 1.0,
		EvalGames:       1,
		WeightStd:       0.1,
		HiddenSize:      3,
	}
	g := NewGenome(cfg)
	pNet, vNet := g.ToNetworks()

	pw := pNet.GetWeights()
	if len(pw) != len(g.PolicyWeights) {
		t.Fatalf("policy weights length mismatch: got %d, want %d", len(pw), len(g.PolicyWeights))
	}
	if !reflect.DeepEqual(pw, g.PolicyWeights) {
		t.Error("policy weights do not round-trip")
	}

	vw := vNet.GetWeights()
	if len(vw) != len(g.ValueWeights) {
		t.Fatalf("value weights length mismatch: got %d, want %d", len(vw), len(g.ValueWeights))
	}
	if !reflect.DeepEqual(vw, g.ValueWeights) {
		t.Error("value weights do not round-trip")
	}
}
