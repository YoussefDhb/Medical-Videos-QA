"""
compare_search_methods.py

Compare pure dense search vs hybrid search to analyze improvements.

Usage:
    python compare_search_methods.py --query "How to do a mouth cancer check at home?" --expected_video "_6csIJAWj_s"
"""

import argparse
import subprocess
import json
import os
import glob

def discover_index(base_dir, pattern):
    """Recursively search for FAISS index files in base_dir matching pattern"""
    search_pattern = os.path.join(base_dir, '**', pattern)
    matches = glob.glob(search_pattern, recursive=True)
    return matches[0] if matches else None

def run_search(query, text_index, visual_index, json_dir, top_k, split, hybrid=False, alpha=0.7):
    """Run search and return results"""
    # Use different output files for dense vs hybrid
    output_file = f'multimodal_search_results_{"hybrid" if hybrid else "dense"}.json'

    cmd = [
        'python', 'query_faiss.py',
        '--query', query,
        '--split', split,
        '--text_index', text_index,
        '--visual_index', visual_index,
        '--json_dir', json_dir,
        '--final_k', str(top_k),
        '--mode', 'segment',
        '--hierarchical',
        '--output', output_file
    ]

    if hybrid:
        cmd.extend(['--hybrid', '--alpha', str(alpha), '--fusion', 'linear', '--expand_query'])

    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print any errors
    if result.returncode != 0:
        print(f"Error running search:")
        print(result.stderr)
        return None

    # Load results from unique output file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Output file not found: {output_file}")
        print(f"Command: {' '.join(cmd)}")
        if result.stdout:
            print(f"stdout: {result.stdout[:500]}")

    return None

def analyze_results(results, expected_video_id):
    """Analyze search results and find position of expected video"""
    if not results or 'results' not in results:
        return None, []

    video_positions = {}
    top_results = []

    for rank, result in enumerate(results['results'], 1):
        video_id = result.get('video_id', 'unknown')
        score = result.get('combined_score', 0.0)

        top_results.append({
            'rank': rank,
            'video_id': video_id,
            'score': score,
            'segment_id': result.get('segment_id', 'unknown')
        })

        if video_id not in video_positions:
            video_positions[video_id] = rank

    expected_rank = video_positions.get(expected_video_id, None)

    return expected_rank, top_results

def main():
    parser = argparse.ArgumentParser(description="Compare pure dense vs hybrid search")
    parser.add_argument("--query", type=str, default="How to do a mouth cancer check at home?",
                       help="Test query")
    parser.add_argument("--expected_video", type=str, default="_6csIJAWj_s",
                       help="Expected video ID for this query")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "test", "val", "all"],
                       help="Split to use (auto-inferred from index name if not provided)")
    parser.add_argument("--text_index", type=str, default=None,
                       help="Path to textual FAISS index (auto-discovers in --faiss_base_dir if not provided)")
    parser.add_argument("--visual_index", type=str, default=None,
                       help="Path to visual FAISS index (auto-discovers in --faiss_base_dir if not provided)")
    parser.add_argument("--json_dir", type=str, default="feature_extraction/",
                       help="Directory with JSON feature files (searches recursively through subfolders)")
    parser.add_argument("--faiss_base_dir", type=str, default="faiss_db",
                       help="Base directory to search for FAISS indices recursively (default: faiss_db)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of results to analyze")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Dense weight for hybrid search")

    args = parser.parse_args()

    # Auto-discover indices if not explicitly provided
    if not args.text_index:
        args.text_index = discover_index(args.faiss_base_dir, 'textual_*.index')
        if not args.text_index:
            print(f"Could not find textual_*.index in {args.faiss_base_dir}")
            return
        print(f"Auto-discovered textual index: {args.text_index}")

    if not args.visual_index:
        args.visual_index = discover_index(args.faiss_base_dir, 'visual_*.index')
        if not args.visual_index:
            print(f"Could not find visual_*.index in {args.faiss_base_dir}")
            return
        print(f"Auto-discovered visual index: {args.visual_index}")

    # Infer split if not provided
    if not args.split:
        for s in ["train", "test", "val"]:
            if s in args.text_index.lower() or s in args.visual_index.lower():
                args.split = s
                print(f"Inferred split: {args.split}")
                break
        
        if not args.split:
            args.split = "test"  # Default to test
            print(f"Could not infer split from index names. Defaulting to: {args.split}")

    print("="*100)
    print("SEARCH METHOD COMPARISON")
    print("="*100)
    print(f"\nQuery: {args.query}")
    print(f"Expected Video: {args.expected_video}")
    print(f"Split: {args.split}")
    print(f"Top-K: {args.top_k}")
    print("\n" + "="*100)

    # Run pure dense search
    print("\n[1/2] Running PURE DENSE search...")
    dense_results = run_search(
        args.query, args.text_index, args.visual_index,
        args.json_dir, args.top_k, args.split, hybrid=False
    )

    dense_rank, dense_top = analyze_results(dense_results, args.expected_video)

    # Run hybrid search
    print("\n[2/2] Running HYBRID search (BM25 + Dense)...")
    hybrid_results = run_search(
        args.query, args.text_index, args.visual_index,
        args.json_dir, args.top_k, args.split, hybrid=True, alpha=args.alpha
    )

    hybrid_rank, hybrid_top = analyze_results(hybrid_results, args.expected_video)

    # Display comparison
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    print(f"\n📍 Expected Video: {args.expected_video}")
    print(f"\n🔍 Pure Dense Search:")
    if dense_rank:
        print(f"   Rank: #{dense_rank} / {args.top_k}")
        print(f"   Found in top-{args.top_k}")
    else:
        print(f"   Not found in top-{args.top_k}")

    print(f"\\nHybrid Search (BM25 + Dense, α={args.alpha}):")
    if hybrid_rank:
        print(f"   Rank: #{hybrid_rank} / {args.top_k}")
        print(f"   Found in top-{args.top_k}")
    else:
        print(f"   Not found in top-{args.top_k}")

    # Calculate improvement
    if dense_rank and hybrid_rank:
        improvement = dense_rank - hybrid_rank
        if improvement > 0:
            print(f"\nIMPROVEMENT: Moved up {improvement} positions!")
        elif improvement < 0:
            print(f"\nREGRESSION: Moved down {abs(improvement)} positions")
        else:
            print(f"\n➡️  NO CHANGE: Same ranking")
    elif hybrid_rank and not dense_rank:
        print(f"\nSUCCESS: Expected video now appears in top-{args.top_k}!")
    elif dense_rank and not hybrid_rank:
        print(f"\nWARNING: Expected video dropped out of top-{args.top_k}")

    # Display top results side by side
    print(f"\n" + "="*100)
    print(f"TOP-{min(5, args.top_k)} RESULTS COMPARISON")
    print("="*100)

    print(f"\n{'RANK':<6} {'DENSE SEARCH':<40} {'HYBRID SEARCH':<40}")
    print("-" * 100)

    for i in range(min(5, args.top_k)):
        rank = i + 1

        dense_video = dense_top[i]['video_id'] if i < len(dense_top) else 'N/A'
        dense_score = f"{dense_top[i]['score']:.4f}" if i < len(dense_top) else 'N/A'

        hybrid_video = hybrid_top[i]['video_id'] if i < len(hybrid_top) else 'N/A'
        hybrid_score = f"{hybrid_top[i]['score']:.4f}" if i < len(hybrid_top) else 'N/A'

        # Highlight expected video
        if dense_video == args.expected_video:
            dense_display = f"{dense_video} ({dense_score})"
        else:
            dense_display = f"   {dense_video} ({dense_score})"

        if hybrid_video == args.expected_video:
            hybrid_display = f"{hybrid_video} ({hybrid_score})"
        else:
            hybrid_display = f"   {hybrid_video} ({hybrid_score})"

        print(f"{rank:<6} {dense_display:<40} {hybrid_display:<40}")

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    if hybrid_rank and (not dense_rank or hybrid_rank < dense_rank):
        print("\nHybrid search shows improvement!")
        print("\nNext steps:")
        print("1. Try different alpha values (0.5-0.9) to find optimal balance")
        print("2. Consider RRF fusion: --fusion rrf")
        print("3. Expand medical terminology dictionary in hybrid_search.py")
    else:
        print("\nHybrid search needs tuning")
        print("\nNext steps:")
        print("1. Reduce alpha (give more weight to BM25): --alpha 0.5")
        print("2. Check query expansion - add more medical synonyms")
        print("3. Consider RRF fusion instead of linear: --fusion rrf")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()
