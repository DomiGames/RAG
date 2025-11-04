package main

import (
	"bufio"
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
	_ "github.com/mattn/go-sqlite3"
	"github.com/ollama/ollama/api"
)

func initDB() {
	err := os.MkdirAll("embeddings", 0755)
	if err != nil {
		log.Fatal("Failed to create directory:", err)
	}

	db, err := sql.Open("sqlite3", "embeddings/store.db?cache=shared&_journal_mode=WAL")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	sqlStmt := `
	CREATE TABLE IF NOT EXISTS chunks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		content TEXT,
		source_file TEXT,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	`
	_, err = db.Exec(sqlStmt)
	if err != nil {
		log.Fatal("Failed to create table:", err)
	}
}

func extractText(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	var text strings.Builder
	reader, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("failed to get PDF text reader: %w", err)
	}
	buf := make([]byte, 1024)

	for {
		n, err := reader.Read(buf)
		if n == 0 || err != nil {
			break
		}
		text.Write(buf[:n])
	}

	return text.String(), nil
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

func saveChunks(chunks []string, sourceFile string) {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("Failed to open database in saveChunks:", err)
	}
	defer db.Close()

	for _, chunk := range chunks {
		_, err = db.Exec("INSERT INTO chunks (content, source_file) VALUES (?, ?)", chunk, sourceFile)
		if err != nil {
			log.Fatal("Failed to insert chunk:", err)
		}
	}
}

func preprocessText(text string) []string {
	text = strings.ToLower(text)
	
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "is": true, "are": true, "was": true, "were": true,
		"be": true, "been": true, "being": true, "have": true, "has": true, "had": true,
		"do": true, "does": true, "did": true, "will": true, "would": true, "could": true,
		"should": true, "may": true, "might": true, "can": true, "this": true, "that": true,
		"these": true, "those": true, "i": true, "you": true, "he": true, "she": true,
		"it": true, "we": true, "they": true, "me": true, "him": true, "her": true,
		"us": true, "them": true, "my": true, "your": true, "his": true, "its": true,
		"our": true, "their": true, "what": true, "which": true, "who": true, "whom": true,
		"when": true, "where": true, "why": true, "how": true, "all": true, "any": true,
		"both": true, "each": true, "few": true, "more": true, "most": true, "other": true,
		"some": true, "such": true, "no": true, "nor": true, "not": true, "only": true,
		"own": true, "same": true, "so": true, "than": true, "too": true, "very": true,
	}
	
	words := strings.Fields(text)
	var filtered []string
	
	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		if len(word) > 2 && !stopWords[word] {
			filtered = append(filtered, word)
		}
	}
	
	return filtered
}

func similarity(a, b string) float64 {
	wordsA := preprocessText(a)
	wordsB := preprocessText(b)
	
	if len(wordsA) == 0 || len(wordsB) == 0 {
		return 0
	}
	
	freqA := make(map[string]int)
	freqB := make(map[string]int)
	
	for _, word := range wordsA {
		freqA[word]++
	}
	for _, word := range wordsB {
		freqB[word]++
	}
	
	dotProduct := 0.0
	magnitudeA := 0.0
	magnitudeB := 0.0
	
	for word, countA := range freqA {
		countB := freqB[word]
		dotProduct += float64(countA * countB)
		magnitudeA += float64(countA * countA)
	}
	
	for _, countB := range freqB {
		magnitudeB += float64(countB * countB)
	}
	
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))
}

func retrieveContext(query string) string {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("Failed to open database in retrieveContext:", err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT content, source_file FROM chunks")
	if err != nil {
		log.Fatal("Failed to query chunks:", err)
	}
	defer rows.Close()

	var bestContent string
	highestScore := -1.0
	queryLower := strings.ToLower(query)
	
	for rows.Next() {
		var content string
		var sourceFile string
		err = rows.Scan(&content, &sourceFile)
		if err != nil {
			continue
		}
		
		score := similarity(query, content)
		
		fileName := strings.ToLower(strings.TrimSuffix(sourceFile, ".pdf"))
		if strings.Contains(queryLower, strings.ReplaceAll(fileName, "_", " ")) ||
		   strings.Contains(queryLower, strings.ReplaceAll(fileName, "-", " ")) {
			score *= 2.0
		}
		
		if score > highestScore {
			highestScore = score
			bestContent = content
		}
	}

	if err = rows.Err(); err != nil {
		log.Fatal("Error iterating over rows:", err)
	}

	return bestContent
}

func askTinyLlama(prompt string) string {
	retrievedContext := retrieveContext(prompt)
	fullPrompt := fmt.Sprintf("Context:\n%s\n\nQuestion:\n%s", retrievedContext, prompt)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal("Failed to create Ollama client:", err)
	}

	req := &api.GenerateRequest{
		Model:  "tinyllama",
		Prompt: fullPrompt,
		Stream: func(b bool) *bool { return &b }(true),
	}

	ctx := context.Background()
	var result strings.Builder

	err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		result.WriteString(resp.Response)
		return nil
	})

	if err != nil {
		log.Fatal("Error during LLM generation:", err)
	}

	return result.String()
}

func ingestFile(path string) error {
	fmt.Printf("Ingesting: %s\n", path)

	text, err := extractText(path)
	if err != nil {
		fmt.Printf("Error processing %s: %v\n", path, err)
		return err
	}

	chunks := chunkText(text, 200)
	saveChunks(chunks, filepath.Base(path))
	fmt.Printf("Saved %d chunks from %s\n", len(chunks), path)
	return nil
}

func ingestFolder(folderPath string) {
	fmt.Printf("Scanning folder: %s\n", folderPath)

	if _, err := os.Stat(folderPath); os.IsNotExist(err) {
		fmt.Printf("Folder does not exist: %s\n", folderPath)
		return
	}

	pdfFiles := []string{}
	err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.ToLower(filepath.Ext(path)) == ".pdf" {
			pdfFiles = append(pdfFiles, path)
		}
		return nil
	})

	if err != nil {
		log.Fatal("Error scanning folder:", err)
	}

	if len(pdfFiles) == 0 {
		fmt.Printf("No PDF files found in: %s\n", folderPath)
		return
	}

	fmt.Printf("Found %d PDF files\n", len(pdfFiles))

	successful := 0
	for i, filePath := range pdfFiles {
		fmt.Printf("[%d/%d] ", i+1, len(pdfFiles))
		if err := ingestFile(filePath); err == nil {
			successful++
		}
	}

	fmt.Printf("Finished: %d/%d PDFs successfully ingested from: %s\n", successful, len(pdfFiles), folderPath)
}

func showIngestionStatus() {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	rows, err := db.Query(`
		SELECT source_file, COUNT(*) as chunk_count 
		FROM chunks 
		GROUP BY source_file
		ORDER BY source_file
	`)
	if err != nil {
		log.Fatal("Failed to query chunks:", err)
	}
	defer rows.Close()

	fmt.Println("Current Database Status:")
	fmt.Println("-----------------------")

	totalChunks := 0
	fileCount := 0

	for rows.Next() {
		var sourceFile string
		var chunkCount int
		err = rows.Scan(&sourceFile, &chunkCount)
		if err != nil {
			log.Fatal("Failed to scan row:", err)
		}
		fmt.Printf("%s: %d chunks\n", sourceFile, chunkCount)
		totalChunks += chunkCount
		fileCount++
	}

	fmt.Println("-----------------------")
	fmt.Printf("Total: %d files, %d chunks\n", fileCount, totalChunks)
}

func clearDatabase() {
	db, err := sql.Open("sqlite3", "embeddings/store.db")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	_, err = db.Exec("DELETE FROM chunks")
	if err != nil {
		log.Fatal("Failed to clear database:", err)
	}

	fmt.Println("Database cleared!")
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

		switch {
		case input == "ingest":
			ingestFolder("documents")

		case strings.HasPrefix(input, "ingest "):
			path := strings.TrimSpace(strings.TrimPrefix(input, "ingest"))
			if path == "" {
				fmt.Println("Please provide a file path. Usage: ingest /path/to/file.pdf")
				continue
			}

			fileInfo, err := os.Stat(path)
			if err != nil {
				fmt.Printf("Path not found: %s\n", path)
				continue
			}

			if fileInfo.IsDir() {
				ingestFolder(path)
			} else {
				ingestFile(path)
			}

		case input == "status":
			showIngestionStatus()

		case input == "clear":
			clearDatabase()

		case input != "":
			answer := askTinyLlama(input)
			fmt.Println("\nAI:", answer)
		}
	}
}
