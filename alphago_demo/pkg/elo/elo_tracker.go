package elo

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// MatchResult represents the outcome of a match between two models
type MatchResult struct {
	Model1     string    `json:"model1"`
	Model2     string    `json:"model2"`
	Result     float64   `json:"result"` // 1.0 = model1 win, 0.5 = draw, 0.0 = model2 win
	NewRating1 float64   `json:"new_rating1"`
	NewRating2 float64   `json:"new_rating2"`
	Timestamp  time.Time `json:"timestamp"`
	GameCount  int       `json:"game_count"` // Number of games in the match
	Comment    string    `json:"comment,omitempty"`
}

// ELOTracker manages ELO ratings for multiple models
type ELOTracker struct {
	BaseRating   float64                  `json:"base_rating"` // Starting rating for new models
	KFactor      float64                  `json:"k_factor"`    // Determines rating change magnitude
	ModelRatings map[string]float64       `json:"model_ratings"`
	MatchHistory []MatchResult            `json:"match_history"`
	ModelMeta    map[string]ModelMetadata `json:"model_metadata"`
	SaveDir      string                   `json:"-"` // Directory to save ratings (not serialized)
}

// ModelMetadata contains descriptive information about a model
type ModelMetadata struct {
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Created     time.Time `json:"created"`
	HiddenSize  int       `json:"hidden_size"`
	SelfGames   int       `json:"self_play_games"`
	Epochs      int       `json:"epochs"`
	Parameters  int       `json:"parameters,omitempty"`
}

// NewELOTracker creates a new ELO rating tracker
func NewELOTracker(baseRating float64, kFactor float64, saveDir string) *ELOTracker {
	return &ELOTracker{
		BaseRating:   baseRating,
		KFactor:      kFactor,
		ModelRatings: make(map[string]float64),
		MatchHistory: make([]MatchResult, 0),
		ModelMeta:    make(map[string]ModelMetadata),
		SaveDir:      saveDir,
	}
}

// RegisterModel adds a new model to the tracker
func (e *ELOTracker) RegisterModel(modelID string, metadata ModelMetadata) float64 {
	// If model already exists, keep its rating
	if _, exists := e.ModelRatings[modelID]; !exists {
		e.ModelRatings[modelID] = e.BaseRating
	}

	// Update or add metadata
	e.ModelMeta[modelID] = metadata

	return e.ModelRatings[modelID]
}

// GetRating returns the ELO rating for a model
func (e *ELOTracker) GetRating(modelID string) float64 {
	rating, exists := e.ModelRatings[modelID]
	if !exists {
		return e.BaseRating
	}
	return rating
}

// ExpectedScore calculates the expected outcome between two models
func (e *ELOTracker) ExpectedScore(model1, model2 string) float64 {
	rating1 := e.GetRating(model1)
	rating2 := e.GetRating(model2)

	// Standard ELO formula for expected outcome
	return 1.0 / (1.0 + math.Pow(10, (rating2-rating1)/400.0))
}

// UpdateRating updates the ELO ratings after a match
func (e *ELOTracker) UpdateRating(model1, model2 string, result float64, gameCount int, comment string) MatchResult {
	// Ensure models exist
	if _, exists := e.ModelRatings[model1]; !exists {
		e.ModelRatings[model1] = e.BaseRating
	}
	if _, exists := e.ModelRatings[model2]; !exists {
		e.ModelRatings[model2] = e.BaseRating
	}

	// Get current ratings
	rating1 := e.ModelRatings[model1]
	rating2 := e.ModelRatings[model2]

	// Calculate expected scores
	expected1 := e.ExpectedScore(model1, model2)
	expected2 := 1.0 - expected1

	// Calculate effective K-factor based on game count
	effectiveK := e.KFactor
	if gameCount > 1 {
		// Scale K-factor by square root of game count for more confidence
		effectiveK *= math.Sqrt(float64(gameCount)) / math.Sqrt(30.0) // Normalize to 30 games

		// Cap the effective K-factor
		if effectiveK > e.KFactor*2 {
			effectiveK = e.KFactor * 2
		}
	}

	// Update ratings
	newRating1 := rating1 + effectiveK*(result-expected1)
	newRating2 := rating2 + effectiveK*((1.0-result)-expected2)

	// Store updated ratings
	e.ModelRatings[model1] = newRating1
	e.ModelRatings[model2] = newRating2

	// Create match result record
	matchResult := MatchResult{
		Model1:     model1,
		Model2:     model2,
		Result:     result,
		NewRating1: newRating1,
		NewRating2: newRating2,
		Timestamp:  time.Now(),
		GameCount:  gameCount,
		Comment:    comment,
	}

	// Add to history
	e.MatchHistory = append(e.MatchHistory, matchResult)

	// Save updated data
	e.Save()

	return matchResult
}

// GetRatingHistory returns the history of rating changes for a model
func (e *ELOTracker) GetRatingHistory(modelID string) []struct {
	Time   time.Time
	Rating float64
} {
	history := make([]struct {
		Time   time.Time
		Rating float64
	}, 0)

	// Add initial rating if model exists
	if _, exists := e.ModelRatings[modelID]; exists {
		// Start with base rating at model creation time or first match
		var startTime time.Time
		if meta, hasMeta := e.ModelMeta[modelID]; hasMeta {
			startTime = meta.Created
		} else if len(e.MatchHistory) > 0 {
			// Find first match involving this model
			for _, match := range e.MatchHistory {
				if match.Model1 == modelID || match.Model2 == modelID {
					startTime = match.Timestamp
					break
				}
			}
		} else {
			startTime = time.Now()
		}

		// Add base rating point
		history = append(history, struct {
			Time   time.Time
			Rating float64
		}{
			Time:   startTime,
			Rating: e.BaseRating,
		})
	}

	// Traverse match history to build rating curve
	for _, match := range e.MatchHistory {
		var rating float64

		if match.Model1 == modelID {
			rating = match.NewRating1
			history = append(history, struct {
				Time   time.Time
				Rating float64
			}{
				Time:   match.Timestamp,
				Rating: rating,
			})
		} else if match.Model2 == modelID {
			rating = match.NewRating2
			history = append(history, struct {
				Time   time.Time
				Rating float64
			}{
				Time:   match.Timestamp,
				Rating: rating,
			})
		}
	}

	// Sort by time
	sort.Slice(history, func(i, j int) bool {
		return history[i].Time.Before(history[j].Time)
	})

	return history
}

// GetTopModels returns the N highest rated models
func (e *ELOTracker) GetTopModels(n int) []struct {
	ModelID string
	Rating  float64
} {
	// Create a slice of model IDs and ratings
	models := make([]struct {
		ModelID string
		Rating  float64
	}, 0, len(e.ModelRatings))

	for id, rating := range e.ModelRatings {
		models = append(models, struct {
			ModelID string
			Rating  float64
		}{
			ModelID: id,
			Rating:  rating,
		})
	}

	// Sort by rating (descending)
	sort.Slice(models, func(i, j int) bool {
		return models[i].Rating > models[j].Rating
	})

	// Return top N (or all if fewer than N)
	if n > len(models) {
		n = len(models)
	}

	return models[:n]
}

// Save writes the ELO tracker data to disk
func (e *ELOTracker) Save() error {
	// Ensure directory exists
	if e.SaveDir == "" {
		e.SaveDir = "elo_ratings"
	}

	err := os.MkdirAll(e.SaveDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create directory %s: %v", e.SaveDir, err)
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal ELO data: %v", err)
	}

	// Write to file
	filename := filepath.Join(e.SaveDir, "elo_ratings.json")
	err = ioutil.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write ELO data to %s: %v", filename, err)
	}

	return nil
}

// Load reads the ELO tracker data from disk
func Load(saveDir string) (*ELOTracker, error) {
	if saveDir == "" {
		saveDir = "elo_ratings"
	}

	filename := filepath.Join(saveDir, "elo_ratings.json")

	// Check if file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		// File doesn't exist, create a new tracker
		return NewELOTracker(1400, 32.0, saveDir), nil
	}

	// Read file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read ELO data from %s: %v", filename, err)
	}

	// Unmarshal from JSON
	var tracker ELOTracker
	err = json.Unmarshal(data, &tracker)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ELO data: %v", err)
	}

	// Set save directory
	tracker.SaveDir = saveDir

	return &tracker, nil
}

// GenerateReport creates a summary report of ELO ratings and match history
func (e *ELOTracker) GenerateReport() string {
	// Get top models sorted by rating
	topModels := e.GetTopModels(len(e.ModelRatings))

	report := "ELO Rating Report\n"
	report += "================\n\n"

	// Display model rankings
	report += fmt.Sprintf("Base rating: %.0f, K-factor: %.1f\n\n", e.BaseRating, e.KFactor)
	report += "Model Rankings:\n"
	report += "--------------\n"

	for i, model := range topModels {
		meta, hasMeta := e.ModelMeta[model.ModelID]
		name := model.ModelID
		if hasMeta {
			name = meta.Name
		}

		report += fmt.Sprintf("%d. %s: %.1f", i+1, name, model.Rating)

		if hasMeta {
			report += fmt.Sprintf(" (Hidden: %d, Games: %d, Epochs: %d)",
				meta.HiddenSize, meta.SelfGames, meta.Epochs)
		}

		report += "\n"
	}

	// Recent matches
	report += "\nRecent Matches:\n"
	report += "--------------\n"

	// Sort matches by time (descending)
	matches := make([]MatchResult, len(e.MatchHistory))
	copy(matches, e.MatchHistory)
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Timestamp.After(matches[j].Timestamp)
	})

	// Show last 10 matches
	numMatches := 10
	if len(matches) < numMatches {
		numMatches = len(matches)
	}

	for i := 0; i < numMatches; i++ {
		match := matches[i]

		// Get model names
		model1Name := match.Model1
		if meta, ok := e.ModelMeta[match.Model1]; ok {
			model1Name = meta.Name
		}

		model2Name := match.Model2
		if meta, ok := e.ModelMeta[match.Model2]; ok {
			model2Name = meta.Name
		}

		// Format result
		var resultStr string
		if match.Result == 1.0 {
			resultStr = "won"
		} else if match.Result == 0.0 {
			resultStr = "lost to"
		} else {
			resultStr = "drew with"
		}

		// Format match details
		report += fmt.Sprintf("%s %s %s (%d games) - [%.1f → %.1f vs %.1f → %.1f] %s\n",
			model1Name, resultStr, model2Name, match.GameCount,
			match.NewRating1-(match.NewRating1-match.NewRating2)/2, match.NewRating1,
			match.NewRating2-(match.NewRating2-match.NewRating1)/2, match.NewRating2,
			match.Timestamp.Format("2006-01-02"))
	}

	return report
}
