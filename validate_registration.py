"""
Validation script for team registration file.

Usage:
    python validate_registration.py [path_to_registration.json]
    
If no path is provided, defaults to teams/team_registration.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def validate_registration(registration_file: str) -> bool:
    """
    Validate team registration file.
    
    Args:
        registration_file: Path to team_registration.json
        
    Returns:
        True if valid, False otherwise
    """
    file_path = Path(registration_file)
    
    print(f"Validating registration file: {file_path}")
    print("=" * 60)
    
    # Check file exists
    if not file_path.exists():
        print(f"❌ ERROR: File not found: {file_path}")
        return False
    
    # Try to parse JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON syntax")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Failed to read file")
        print(f"   {e}")
        return False
    
    print("✓ JSON syntax is valid")
    
    # Check structure
    if "teams" not in data:
        print('❌ ERROR: Missing "teams" key in JSON')
        return False
    
    if not isinstance(data["teams"], list):
        print('❌ ERROR: "teams" must be a list')
        return False
    
    print("✓ JSON structure is valid")
    
    # Validate teams
    teams = data["teams"]
    print(f"✓ Found {len(teams)} team(s)")
    print()
    
    issues = []
    warnings = []
    team_names = []
    all_student_ids = []
    student_id_to_teams = defaultdict(list)
    
    for i, team in enumerate(teams, 1):
        team_name = team.get("team_name")
        members = team.get("members", [])
        
        # Check team name
        if not team_name:
            issues.append(f"Team {i}: Missing team_name")
            continue
        
        if team_name in team_names:
            issues.append(f"Team {i} ({team_name}): Duplicate team name")
        else:
            team_names.append(team_name)
        
        # Check members
        if not isinstance(members, list):
            issues.append(f"Team {team_name}: members must be a list")
            continue
        
        if len(members) == 0:
            warnings.append(f"Team {team_name}: No members registered")
        
        # Track student IDs
        for student_id in members:
            if not isinstance(student_id, str):
                issues.append(f"Team {team_name}: Student ID '{student_id}' must be a string")
            else:
                all_student_ids.append(student_id)
                student_id_to_teams[student_id].append(team_name)
        
        print(f"Team {i}: {team_name}")
        print(f"  - Members: {len(members)}")
        if members:
            print(f"  - IDs: {', '.join(members)}")
    
    print()
    
    # Check for duplicate student IDs across teams
    duplicate_ids = {sid: teams for sid, teams in student_id_to_teams.items() if len(teams) > 1}
    if duplicate_ids:
        for student_id, team_list in duplicate_ids.items():
            issues.append(f"Student ID '{student_id}' appears in multiple teams: {', '.join(team_list)}")
    
    # Print warnings
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    # Print issues
    if issues:
        print("❌ ERRORS:")
        for issue in issues:
            print(f"   {issue}")
        print()
        return False
    
    # Summary
    print("=" * 60)
    print("✓ VALIDATION PASSED")
    print(f"  - {len(team_names)} teams registered")
    print(f"  - {len(all_student_ids)} total student IDs")
    print(f"  - {len(set(all_student_ids))} unique student IDs")
    
    return True


def check_teams_directory(registration_file: str, teams_dir: str = "teams") -> None:
    """
    Check if team folders match registration.
    
    Args:
        registration_file: Path to team_registration.json
        teams_dir: Path to teams directory
    """
    print()
    print("=" * 60)
    print("Checking teams directory...")
    print()
    
    reg_path = Path(registration_file)
    teams_path = Path(teams_dir)
    
    if not teams_path.exists():
        print(f"⚠️  Teams directory not found: {teams_path}")
        return
    
    # Load registration
    try:
        with open(reg_path, 'r') as f:
            data = json.load(f)
        registered_teams = {team["team_name"] for team in data.get("teams", []) if "team_name" in team}
    except:
        print("❌ Could not load registration file")
        return
    
    # Find team folders
    team_folders = {d.name for d in teams_path.iterdir() if d.is_dir()}
    
    # Compare
    registered_not_found = registered_teams - team_folders
    folders_not_registered = team_folders - registered_teams
    
    if registered_not_found:
        print("⚠️  Registered teams without folders:")
        for team in sorted(registered_not_found):
            print(f"   - {team}")
        print()
    
    if folders_not_registered:
        print("⚠️  Team folders without registration:")
        for team in sorted(folders_not_registered):
            print(f"   - {team}")
        print()
    
    if not registered_not_found and not folders_not_registered:
        print("✓ All registered teams have folders")
        print("✓ All team folders are registered")


if __name__ == "__main__":
    # Get registration file path
    if len(sys.argv) > 1:
        registration_file = sys.argv[1]
    else:
        registration_file = "teams/team_registration.json"
    
    # Validate
    is_valid = validate_registration(registration_file)
    
    # Check teams directory if validation passed
    if is_valid:
        # Determine teams directory (same directory as registration file)
        teams_dir = Path(registration_file).parent
        check_teams_directory(registration_file, str(teams_dir))
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)
