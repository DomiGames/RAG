package main

import (
	"bufio"
	"context"  // This import is now used properly
	"database/sql"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ledongthuc/pdf"
	_ "github.com/mattn/go-sqlite3"
	"github.com/ollama/ollama/api"
)

func initDB() {
	err := os.MkdirAll("embeddings", 0755)
	if err != nil {
		log.Fatal("😡 Failed to create directory:", err)
	}

	db, err := sql.Open("sqlite3", "embeddings/store.db?cache=shared&_journal_mode=WAL")
	if err != nil {
		log.Fatal("😡 Failed to open database:", err)
	}
	defer db.Close()

	sqlStmt := `
	CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		content TEXT
	);
	`
	_, err = db.Exec(sqlStmt)
	if err != nil {
		log.Fatal("😡 Failed to create table:", err)
	}
}

func extractText(path string) string {
	f, r, err := pdf.Open(path)
	if err != nil {
		log.Fatal("😡 Failed to open PDF:", err)
	}
	defer f.Close()

	var text strings.Builder
	reader, err := r.GetPlainText()
	if err != nil {
		log.Fatal("😡 Failed to get PDF text reader:", err)
	}
	buf := make([]byte, 1024)

	for {
		n, err := reader.Read(buf)
		if n == 0 || err != nil {
			break
		}
		text.Write(buf[:n])
	}

	return text.String()
}

func chunkText(text string, size int) []string {
	words := strings.Fields(text)
	var chunks []string

	for i := 0; i < len(words); i += size {
		end := i + size
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)
	}
	return chunks
}

func saveChunks(chunks []string) {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("😡 Failed to open database in saveChunks:", err)
	}
	defer db.Close()

	for _, chunk := range chunks {
		_, err = db.Exec("INSERT INTO chunks (content) VALUES (?)", chunk)
		if err != nil {
			log.Fatal("😡 Failed to insert chunk:", err)
		}
	}
}

func retrieveContext(query string) string {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("😡 Failed to open database in retrieveContext:", err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT content FROM chunks")
	if err != nil {
		log.Fatal("😡 Failed to query chunks:", err)
	}
	defer rows.Close()

	var best string
	highest := -1

	for rows.Next() {
		var content string
		err = rows.Scan(&content)
		if err != nil {
			log.Fatal("😡 Failed to scan chunk row:", err)
		}
		score := similarity(query, content)
		if score > highest {
			highest = score
			best = content
		}
	}

	if err = rows.Err(); err != nil {
		log.Fatal("😡 Error iterating over rows:", err)
	}

	return best
}

func similarity(a, b string) int {
	count := 0
	words := strings.Fields(a)
	for _, w := range words {
		if strings.Contains(b, w) {
			count++
		}
	}
	return count
}

func askTinyLlama(prompt string) string {
    // FIXED: Variable renamed to avoid conflict
    retrievedContext := retrieveContext(prompt)
    fullPrompt := fmt.Sprintf("Context:\n%s\n\nQuestion:\n%s", retrievedContext, prompt)

    client, err := api.ClientFromEnvironment()
    if err != nil {
        log.Fatal("😡 Failed to create Ollama client:", err)
    }

    req := &api.GenerateRequest{
        Model:  "tinyllama",
        Prompt: fullPrompt,
        Stream: func(b bool) *bool { return &b }(true),
    }

    // FIXED: context package is no longer shadowed
    ctx := context.Background()
    var result strings.Builder

    err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
        result.WriteString(resp.Response)
        return nil
    })

    if err != nil {
        log.Fatal("😡 Error during LLM generation:", err)
    }

    return result.String()
}

func ingestFile(path string) {
	fmt.Println("📖 Ingesting:", path)
	text := extractText(path)
	chunks := chunkText(text, 200)
	saveChunks(chunks)
	fmt.Printf("✅ Saved %d chunks from %s\n", len(chunks), path)
}

func main() {
	initDB()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()

		if strings.HasPrefix(input, "ingest ") {
			path := strings.TrimSpace(strings.TrimPrefix(input, "ingest"))
			if path == "" {
				fmt.Println("⚠️  Please provide a file path. Usage: ingest /path/to/file.pdf")
				continue
			}
			ingestFile(path)
		} else if input != "" {
			answer := askTinyLlama(input)
			fmt.Println("\n🤖:", answer)
		}
	}
}
