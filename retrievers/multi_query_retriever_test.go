package retrievers

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms/googleai"
)

func TestMultiQueryRetriever(t *testing.T) {
	t.Parallel()
	genaiKey := os.Getenv("GENAI_API_KEY")
	if genaiKey == "" {
		t.Skip("must set GENAI_API_KEY to run test")
	}
	pgvectorURL := os.Getenv("PGVECTOR_CONNECTION_STRING")
	if pgvectorURL == "" {
		t.Skip("Must set PGVECTOR_CONNECTION_STRING to run test")
	}
	ctx := context.Background()

	llm, err := googleai.New(ctx, googleai.WithAPIKey(genaiKey))
	require.NoError(t, err)
	// TBD
}
