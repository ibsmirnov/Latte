import zstandard
import json
import random
import pandas as pd

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	"""
	Decode a chunk of bytes from a zstandard compressed stream.
	Handles UnicodeDecodeError by reading additional chunks if needed.
	"""
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		print(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
	"""
	Generator that yields lines from a zstandard compressed file.
	Each yield returns a line and the current position in the file.
	"""
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)

			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]

		reader.close()
		
def extract(file_name):
	"""
	Extract title and text from Reddit submissions.
	Skips deleted and removed posts, and cleans text by removing extra whitespace.
	Returns a list of (title, text) tuples.
	"""
	results = []

	for line, file_bytes_processed in read_lines_zst(file_name):
		obj = json.loads(line)
		if obj['selftext'] == '[deleted]': continue
		if obj['selftext'] == '[removed]': continue
		text = obj['selftext']
		text = text.replace("\n", " ").replace("\r", " ")
		text = ' '.join(text.split())
		results.append((obj['title'], text))
	return results

# Extract data from compressed files
men_data = extract('data/AskMen_submissions.zst')
women_data = extract('data/AskWomen_submissions.zst')

# Set random seed for reproducibility
random.seed(1)

# Sample equal number of posts from each subreddit
N = 50000
sub_men = random.sample(men_data, N)
sub_women = random.sample(women_data, N)

# Save data to CSV file using pandas
output_file = 'data/reddit_sample.csv'

# Create data for women (gender=0)
women_df = pd.DataFrame(sub_women, columns=['title', 'text'])
women_df['gender'] = 0

# Create data for men (gender=1)
men_df = pd.DataFrame(sub_men, columns=['title', 'text'])
men_df['gender'] = 1

# Combine and save
combined_df = pd.concat([women_df, men_df], ignore_index=True)
combined_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 means QUOTE_ALL

print(f"Successfully saved {len(combined_df)} records to {output_file}")