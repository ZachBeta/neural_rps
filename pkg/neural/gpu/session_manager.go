//go:build gpu
// +build gpu

package gpu

import (
	"fmt"
	"sync"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// SessionManager maintains TensorFlow sessions
type SessionManager struct {
	sessions map[string]*tf.Session
	mu       sync.Mutex
}

// NewSessionManager creates a new session manager
func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*tf.Session),
	}
}

// GetSession retrieves or creates a session for the given model
func (sm *SessionManager) GetSession(modelID string, graphDef []byte) (*tf.Session, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check if session exists
	if session, ok := sm.sessions[modelID]; ok {
		return session, nil
	}

	// Create new graph and session
	graph := tf.NewGraph()
	if err := graph.Import(graphDef, ""); err != nil {
		return nil, fmt.Errorf("error importing graph: %v", err)
	}

	sessionOpts := &tf.SessionOptions{
		Config: []byte(`
			allow_soft_placement: true
			gpu_options { 
				allow_growth: true
				per_process_gpu_memory_fraction: 0.5
			}
		`),
	}

	session, err := tf.NewSession(graph, sessionOpts)
	if err != nil {
		return nil, fmt.Errorf("error creating session: %v", err)
	}

	// Store session
	sm.sessions[modelID] = session
	return session, nil
}

// CloseAll closes all sessions
func (sm *SessionManager) CloseAll() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, session := range sm.sessions {
		session.Close()
	}

	sm.sessions = make(map[string]*tf.Session)
}
