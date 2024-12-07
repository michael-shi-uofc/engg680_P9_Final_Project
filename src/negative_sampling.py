import pandas as pd
import random
import Geohash

def generate_negative_samples(df, ratio=3):
    """
    Generate negative samples by adjusting temporal or spatial features.
    """
    print("Generating negative samples...")
    negative_samples = []
    target_count = len(df) * ratio  # Desired number of negative samples
    max_attempts = 100000  # Limit attempts to avoid infinite loops
    attempts = 0

    while len(negative_samples) < target_count and attempts < max_attempts:
        for _, row in df.iterrows():
            neg_sample = row.copy()

            # Randomly adjust time or location
            if random.choice(['time', 'location']) == 'time':
                choice = random.choice(['day', 'hour', 'month'])
                if choice == 'day':
                    neg_sample['day'] = (row['day'] + random.randint(-2, 2)) % 7  # Monday=0, Sunday=6
                elif choice == 'hour':
                    neg_sample['hour'] = (row['hour'] + random.randint(-3, 3)) % 24
                elif choice == 'month':
                    neg_sample['month'] = (row['month'] + random.randint(-1, 1)) % 12 or 12  # Ensure month is 1-12
            else:
                if random.choice(['randomGeohash', 'existingGeohash']) == 'existingGeohash':
                    neg_sample['geohash'] = random.choice(df['geohash'].unique())
                else:
                    neg_sample['latitude'] += random.uniform(-0.01, 0.01)
                    neg_sample['longitude'] += random.uniform(-0.01, 0.01)
                    neg_sample['geohash'] = Geohash.encode(
                        neg_sample['latitude'], neg_sample['longitude'], precision=5)

            # Add to negative samples only if it doesn't exist as a positive
            if not ((df['geohash'] == neg_sample['geohash']) & 
                    (df['day'] == neg_sample['day']) & 
                    (df['hour'] == neg_sample['hour'])).any():
                negative_samples.append(neg_sample)

                # Check if we've reached the target count
                if len(negative_samples) >= target_count:
                    break
        attempts += 1

    neg_df = pd.DataFrame(negative_samples)
    neg_df['label'] = 0  # Negative samples
    print(f"Generated {len(negative_samples)} negative samples after {attempts} attempts.")
    return neg_df