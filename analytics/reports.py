# analytics/reports.py
"""
Stakeholder Reports and Compliance Certificates.

This module provides:
1. ClubReport - Per-club summary reports for stakeholder review
2. TeamReport - Per-team season schedule
3. GradeReport - Grade-level analysis
4. ComplianceCertificate - Formal compliance verification document
5. HTML report generation for visual presentation

Usage:
    from analytics.reports import ClubReport, ComplianceCertificate
    
    # Generate club report
    report = ClubReport(draw, data)
    report.generate_for_club('Maitland', output='maitland_report.xlsx')
    
    # Generate compliance certificate
    cert = ComplianceCertificate(draw, data)
    cert.generate(output='compliance.xlsx')
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.storage import DrawStorage, StoredGame, DrawAnalytics
from analytics.tester import DrawTester, ViolationReport, Violation


# ============== Club Report ==============

class ClubReport:
    """
    Generate comprehensive club-level reports for stakeholder review.
    
    Each report includes:
    - Club summary (total games, teams, grades)
    - Per-team season schedule
    - Home/away balance analysis
    - Opponent distribution
    - Special events (club days, etc.)
    - Constraint compliance for club's games
    """
    
    def __init__(self, draw: DrawStorage, data: Dict):
        self.draw = draw
        self.data = data
        self.teams = data.get('teams', [])
        self.clubs = data.get('clubs', [])
        self.grades = data.get('grades', [])
        
        # Build lookups
        self._team_to_club = {t.name: t.club.name for t in self.teams}
        self._team_to_grade = {t.name: t.grade for t in self.teams}
    
    def get_club_teams(self, club_name: str) -> List[str]:
        """Get all team names belonging to a club."""
        return [t.name for t in self.teams if t.club.name == club_name]
    
    def get_club_games(self, club_name: str) -> List[StoredGame]:
        """Get all games involving a club's teams."""
        club_teams = self.get_club_teams(club_name)
        return [g for g in self.draw.games 
                if g.team1 in club_teams or g.team2 in club_teams]
    
    def club_summary(self, club_name: str) -> Dict[str, Any]:
        """Generate summary statistics for a club."""
        club_teams = self.get_club_teams(club_name)
        club_games = self.get_club_games(club_name)
        
        # Get club object for home field info
        club_obj = next((c for c in self.clubs if c.name == club_name), None)
        home_field = club_obj.home_field if club_obj else "Unknown"
        
        # Count home/away games
        home_count = 0
        away_count = 0
        neutral_count = 0
        
        for game in club_games:
            if game.field_location == home_field:
                home_count += 1
            elif self._is_neutral_venue(game.field_location):
                neutral_count += 1
            else:
                away_count += 1
        
        # Grade distribution
        grades_playing = set()
        for team in club_teams:
            grade = self._team_to_grade.get(team)
            if grade:
                grades_playing.add(grade)
        
        # Opponent analysis
        opponents = defaultdict(int)
        for game in club_games:
            for team in [game.team1, game.team2]:
                if team not in club_teams:
                    opp_club = self._team_to_club.get(team)
                    if opp_club:
                        opponents[opp_club] += 1
        
        return {
            'club_name': club_name,
            'home_field': home_field,
            'num_teams': len(club_teams),
            'num_grades': len(grades_playing),
            'grades': sorted(grades_playing),
            'total_games': len(club_games),
            'home_games': home_count,
            'away_games': away_count,
            'neutral_games': neutral_count,
            'opponents': dict(opponents),
            'teams': club_teams,
        }
    
    def _is_neutral_venue(self, location: str) -> bool:
        """Check if a venue is neutral (Broadmeadow)."""
        return location == 'Newcastle International Hockey Centre'
    
    def team_schedule(self, team_name: str) -> pd.DataFrame:
        """Generate complete season schedule for a team."""
        team_games = [g for g in self.draw.games 
                     if team_name in (g.team1, g.team2)]
        
        df_data = []
        for game in sorted(team_games, key=lambda g: (g.week, g.day_slot)):
            is_home = game.team1 == team_name
            opponent = game.team2 if is_home else game.team1
            
            df_data.append({
                'Week': game.week,
                'Round': game.round_no,
                'Date': game.date,
                'Day': game.day,
                'Time': game.time,
                'H/A': 'H' if is_home else 'A',
                'Opponent': opponent,
                'Opp Club': self._team_to_club.get(opponent, 'Unknown'),
                'Field': game.field_name,
                'Location': game.field_location,
                'Grade': game.grade
            })
        
        return pd.DataFrame(df_data)
    
    def home_away_by_opponent(self, team_name: str) -> pd.DataFrame:
        """Analyze home/away split per opponent for a team."""
        team_games = [g for g in self.draw.games 
                     if team_name in (g.team1, g.team2)]
        
        # Get team's club and home field
        team_club = self._team_to_club.get(team_name)
        team_obj = next((t for t in self.teams if t.name == team_name), None)
        club_obj = next((c for c in self.clubs if c.name == team_club), None)
        home_field = club_obj.home_field if club_obj else None
        
        opponent_stats = defaultdict(lambda: {'home': 0, 'away': 0, 'neutral': 0})
        
        for game in team_games:
            opponent = game.team2 if game.team1 == team_name else game.team1
            opp_club = self._team_to_club.get(opponent)
            
            # Skip intra-club games
            if opp_club == team_club:
                continue
            
            if home_field and game.field_location == home_field:
                opponent_stats[opponent]['home'] += 1
            elif self._is_neutral_venue(game.field_location):
                opponent_stats[opponent]['neutral'] += 1
            else:
                opponent_stats[opponent]['away'] += 1
        
        df_data = []
        for opponent, stats in sorted(opponent_stats.items()):
            total = stats['home'] + stats['away'] + stats['neutral']
            df_data.append({
                'Opponent': opponent,
                'Opp Club': self._team_to_club.get(opponent),
                'Home': stats['home'],
                'Away': stats['away'],
                'Neutral': stats['neutral'],
                'Total': total,
                'Home%': f"{100*stats['home']/total:.0f}%" if total > 0 else "N/A"
            })
        
        return pd.DataFrame(df_data)
    
    def bye_weeks(self, team_name: str) -> List[int]:
        """Get weeks where a team has a bye."""
        team_games = [g for g in self.draw.games 
                     if team_name in (g.team1, g.team2)]
        
        played_weeks = {g.week for g in team_games}
        all_weeks = {g.week for g in self.draw.games}
        
        return sorted(all_weeks - played_weeks)
    
    def generate_for_club(self, club_name: str, output: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate complete club report.
        
        Args:
            club_name: Name of the club
            output: Optional output Excel file path
            
        Returns:
            Dict mapping sheet names to DataFrames
        """
        sheets = {}
        
        # 1. Club Summary
        summary = self.club_summary(club_name)
        summary_df = pd.DataFrame([{
            'Metric': k,
            'Value': str(v) if not isinstance(v, (int, float)) else v
        } for k, v in summary.items() if k != 'opponents'])
        sheets['Club Summary'] = summary_df
        
        # Opponent distribution
        opp_df = pd.DataFrame([
            {'Opponent Club': k, 'Games Against': v}
            for k, v in sorted(summary['opponents'].items(), key=lambda x: -x[1])
        ])
        sheets['Opponents'] = opp_df
        
        # 2. Team schedules
        club_teams = self.get_club_teams(club_name)
        for team in sorted(club_teams):
            schedule = self.team_schedule(team)
            # Truncate sheet name to Excel limit
            sheet_name = f"Schedule-{team}"[:31]
            sheets[sheet_name] = schedule
        
        # 3. Home/away by opponent for each team
        ha_data = []
        for team in club_teams:
            ha_df = self.home_away_by_opponent(team)
            ha_df['Team'] = team
            ha_data.append(ha_df)
        
        if ha_data:
            combined_ha = pd.concat(ha_data, ignore_index=True)
            sheets['Home-Away Analysis'] = combined_ha
        
        # 4. Bye weeks
        bye_data = []
        for team in club_teams:
            byes = self.bye_weeks(team)
            for week in byes:
                bye_data.append({'Team': team, 'Bye Week': week})
        
        if bye_data:
            sheets['Bye Weeks'] = pd.DataFrame(bye_data)
        
        # Export if requested
        if output:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Club report for '{club_name}' exported to {output}")
        
        return sheets
    
    def generate_all_clubs(self, output_dir: str = "reports") -> None:
        """Generate reports for all clubs."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for club in self.clubs:
            output_path = Path(output_dir) / f"{club.name}_report.xlsx"
            self.generate_for_club(club.name, str(output_path))
        
        print(f"Generated reports for {len(self.clubs)} clubs in {output_dir}/")


# ============== Team Report ==============

class TeamReport:
    """Generate individual team reports."""
    
    def __init__(self, draw: DrawStorage, data: Dict):
        self.draw = draw
        self.data = data
        self.club_report = ClubReport(draw, data)
    
    def generate(self, team_name: str, output: str = None) -> Dict[str, pd.DataFrame]:
        """Generate report for a single team."""
        sheets = {}
        
        # Season schedule
        schedule = self.club_report.team_schedule(team_name)
        sheets['Schedule'] = schedule
        
        # Home/away analysis
        ha = self.club_report.home_away_by_opponent(team_name)
        sheets['Home-Away'] = ha
        
        # Bye weeks
        byes = self.club_report.bye_weeks(team_name)
        sheets['Byes'] = pd.DataFrame({'Bye Week': byes})
        
        # Summary stats
        total_games = len(schedule)
        home_games = len(schedule[schedule['H/A'] == 'H'])
        away_games = len(schedule[schedule['H/A'] == 'A'])
        
        summary = pd.DataFrame([
            {'Metric': 'Total Games', 'Value': total_games},
            {'Metric': 'Home Games', 'Value': home_games},
            {'Metric': 'Away Games', 'Value': away_games},
            {'Metric': 'Bye Weeks', 'Value': len(byes)},
            {'Metric': 'Home %', 'Value': f"{100*home_games/total_games:.1f}%" if total_games > 0 else "N/A"},
        ])
        sheets['Summary'] = summary
        
        if output:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Team report for '{team_name}' exported to {output}")
        
        return sheets


# ============== Grade Report ==============

class GradeReport:
    """Generate grade-level reports showing all games in a grade."""
    
    def __init__(self, draw: DrawStorage, data: Dict):
        self.draw = draw
        self.data = data
        self.grades = data.get('grades', [])
        self.teams = data.get('teams', [])
        self._team_to_club = {t.name: t.club.name for t in self.teams}
    
    def grade_schedule(self, grade_name: str) -> pd.DataFrame:
        """Get complete schedule for a grade."""
        grade_games = [g for g in self.draw.games if g.grade == grade_name]
        
        df_data = []
        for game in sorted(grade_games, key=lambda g: (g.week, g.day_slot)):
            df_data.append({
                'Week': game.week,
                'Round': game.round_no,
                'Date': game.date,
                'Day': game.day,
                'Time': game.time,
                'Team 1': game.team1,
                'Club 1': self._team_to_club.get(game.team1),
                'Team 2': game.team2,
                'Club 2': self._team_to_club.get(game.team2),
                'Field': game.field_name,
                'Location': game.field_location
            })
        
        return pd.DataFrame(df_data)
    
    def matchup_matrix(self, grade_name: str) -> pd.DataFrame:
        """Get matchup matrix for a grade."""
        analytics = DrawAnalytics(self.draw, self.data)
        matrices = analytics.team_matchups_crosstab(grade=grade_name)
        return matrices.get(grade_name, pd.DataFrame())
    
    def generate(self, grade_name: str, output: str = None) -> Dict[str, pd.DataFrame]:
        """Generate report for a grade."""
        sheets = {}
        
        sheets['Schedule'] = self.grade_schedule(grade_name)
        sheets['Matchup Matrix'] = self.matchup_matrix(grade_name)
        
        # Games per team
        team_counts = defaultdict(int)
        for game in self.draw.games:
            if game.grade == grade_name:
                team_counts[game.team1] += 1
                team_counts[game.team2] += 1
        
        sheets['Games Per Team'] = pd.DataFrame([
            {'Team': team, 'Club': self._team_to_club.get(team), 'Games': count}
            for team, count in sorted(team_counts.items())
        ])
        
        if output:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Grade report for '{grade_name}' exported to {output}")
        
        return sheets
    
    def generate_all_grades(self, output_dir: str = "reports") -> None:
        """Generate reports for all grades."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for grade in self.grades:
            output_path = Path(output_dir) / f"Grade_{grade.name}_report.xlsx"
            self.generate(grade.name, str(output_path))
        
        print(f"Generated reports for {len(self.grades)} grades in {output_dir}/")


# ============== Compliance Certificate ==============

class ComplianceCertificate:
    """
    Generate formal compliance verification document.
    
    Shows pass/fail status for each constraint with details.
    Designed for stakeholder review and approval.
    """
    
    CONSTRAINT_DESCRIPTIONS = {
        'NoDoubleBookingTeams': 'No team plays more than once per week',
        'NoDoubleBookingFields': 'No field hosts multiple games at same time',
        'EqualGames': 'Each team plays the expected number of games',
        'BalancedMatchups': 'Each pair of teams meets a balanced number of times',
        'FiftyFiftyHomeAway': 'Away teams (Maitland/Gosford) have balanced home/away',
        'MaxMaitlandHomeWeekends': 'No back-to-back Maitland home weekends',
        'AwayAtMaitlandGrouping': 'Maximum 3 away clubs at Maitland per week',
        'ClubGradeAdjacency': 'Adjacent grades from same club dont clash',
        'PHLAndSecondGradeAdjacency': 'PHL and 2nd grade from same club play nearby',
        'PHLAndSecondGradeTimes': 'PHL plays before or with 2nd grade',
    }
    
    SEVERITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    
    def __init__(self, draw: DrawStorage, data: Dict):
        self.draw = draw
        self.data = data
        self.tester = DrawTester(draw, data)
        self.report: Optional[ViolationReport] = None
    
    def run_checks(self) -> ViolationReport:
        """Run all constraint checks."""
        self.report = self.tester.run_violation_check()
        return self.report
    
    def summary_table(self) -> pd.DataFrame:
        """Generate summary table of all constraints."""
        if self.report is None:
            self.run_checks()
        
        # Group violations by constraint
        violations_by_constraint = self.report.by_constraint()
        
        rows = []
        for constraint, description in self.CONSTRAINT_DESCRIPTIONS.items():
            violations = violations_by_constraint.get(constraint, [])
            
            if not violations:
                status = '✅ PASS'
                severity = '-'
                count = 0
                sample = '-'
            else:
                status = '❌ FAIL'
                severity = violations[0].severity
                count = len(violations)
                sample = violations[0].message[:50] + '...' if len(violations[0].message) > 50 else violations[0].message
            
            rows.append({
                'Constraint': constraint,
                'Description': description,
                'Status': status,
                'Severity': severity,
                'Violations': count,
                'Sample Issue': sample
            })
        
        return pd.DataFrame(rows)
    
    def detailed_violations(self) -> pd.DataFrame:
        """Get detailed list of all violations."""
        if self.report is None:
            self.run_checks()
        
        rows = []
        for v in self.report.violations:
            rows.append({
                'Severity': v.severity,
                'Constraint': v.constraint,
                'Message': v.message,
                'Week': v.week or '-',
                'Affected Games': ', '.join(v.affected_games[:3]) if v.affected_games else '-'
            })
        
        # Sort by severity
        severity_order = {s: i for i, s in enumerate(self.SEVERITY_ORDER)}
        rows.sort(key=lambda r: severity_order.get(r['Severity'], 99))
        
        return pd.DataFrame(rows)
    
    def certificate_header(self) -> pd.DataFrame:
        """Generate certificate header information."""
        if self.report is None:
            self.run_checks()
        
        overall_status = '✅ COMPLIANT' if not self.report.has_violations else '❌ NON-COMPLIANT'
        
        header_data = [
            {'Field': 'Certificate', 'Value': 'Draw Compliance Verification'},
            {'Field': 'Generated', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M')},
            {'Field': 'Draw Description', 'Value': self.draw.description or 'No description'},
            {'Field': 'Total Games', 'Value': self.draw.num_games},
            {'Field': 'Total Weeks', 'Value': self.draw.num_weeks},
            {'Field': 'Overall Status', 'Value': overall_status},
            {'Field': 'Total Violations', 'Value': len(self.report.violations)},
            {'Field': 'Critical Issues', 'Value': self.report.critical_count},
            {'Field': 'High Issues', 'Value': self.report.high_count},
            {'Field': 'Medium Issues', 'Value': self.report.medium_count},
            {'Field': 'Low Issues', 'Value': self.report.low_count},
        ]
        
        return pd.DataFrame(header_data)
    
    def generate(self, output: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate complete compliance certificate.
        
        Args:
            output: Optional output Excel file path
            
        Returns:
            Dict mapping sheet names to DataFrames
        """
        if self.report is None:
            self.run_checks()
        
        sheets = {}
        
        # Certificate header
        sheets['Certificate'] = self.certificate_header()
        
        # Constraint summary
        sheets['Constraint Summary'] = self.summary_table()
        
        # Detailed violations (if any)
        if self.report.has_violations:
            sheets['Violations Detail'] = self.detailed_violations()
        
        if output:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add formatting
                    worksheet = writer.sheets[sheet_name]
                    
                    # Widen columns
                    for i, col in enumerate(df.columns):
                        max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
            
            print(f"Compliance certificate exported to {output}")
        
        return sheets
    
    def is_compliant(self) -> bool:
        """Check if draw is fully compliant (no violations)."""
        if self.report is None:
            self.run_checks()
        return not self.report.has_violations
    
    def is_critical_compliant(self) -> bool:
        """Check if draw has no critical violations."""
        if self.report is None:
            self.run_checks()
        return self.report.critical_count == 0


# ============== HTML Report Generation ==============

def generate_html_report(draw: DrawStorage, data: Dict, output: str = "draw_report.html") -> None:
    """
    Generate an HTML report for visual presentation.
    
    Args:
        draw: DrawStorage object
        data: Data dictionary
        output: Output HTML file path
    """
    # Run compliance check
    cert = ComplianceCertificate(draw, data)
    cert.run_checks()
    
    # Get analytics
    analytics = DrawAnalytics(draw, data)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Draw Report - {draw.description or 'Season Schedule'}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a5f, #2563eb);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .status-pass {{
            background: #10b981;
            color: white;
        }}
        .status-fail {{
            background: #ef4444;
            color: white;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
        }}
        .severity-critical {{ color: #dc2626; font-weight: bold; }}
        .severity-high {{ color: #ea580c; font-weight: bold; }}
        .severity-medium {{ color: #ca8a04; }}
        .severity-low {{ color: #65a30d; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1e3a5f;
        }}
        .stat-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏑 Draw Report</h1>
        <p>{draw.description or 'Hockey Season Schedule'}</p>
        <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        <span class="status-badge {'status-pass' if cert.is_compliant() else 'status-fail'}">
            {'✅ COMPLIANT' if cert.is_compliant() else '❌ NON-COMPLIANT'}
        </span>
    </div>
    
    <div class="card">
        <h2>📊 Summary Statistics</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{draw.num_games}</div>
                <div class="stat-label">Total Games</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{draw.num_weeks}</div>
                <div class="stat-label">Weeks</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(data.get('teams', []))}</div>
                <div class="stat-label">Teams</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(data.get('clubs', []))}</div>
                <div class="stat-label">Clubs</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(data.get('grades', []))}</div>
                <div class="stat-label">Grades</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(cert.report.violations)}</div>
                <div class="stat-label">Violations</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>✅ Constraint Compliance</h2>
        <table>
            <thead>
                <tr>
                    <th>Constraint</th>
                    <th>Status</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
"""
    
    summary_df = cert.summary_table()
    for _, row in summary_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['Constraint']}<br><small style="color:#6b7280">{row['Description']}</small></td>
                    <td>{row['Status']}</td>
                    <td>{row['Violations']}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
"""
    
    # Violations detail if any
    if cert.report.has_violations:
        html_content += """
    <div class="card">
        <h2>⚠️ Violation Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>Constraint</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""
        for v in cert.report.violations[:20]:  # Limit to 20
            severity_class = f"severity-{v.severity.lower()}"
            html_content += f"""
                <tr>
                    <td class="{severity_class}">{v.severity}</td>
                    <td>{v.constraint}</td>
                    <td>{v.message}</td>
                </tr>
"""
        
        if len(cert.report.violations) > 20:
            html_content += f"""
                <tr>
                    <td colspan="3"><em>... and {len(cert.report.violations) - 20} more violations</em></td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output}")


# ============== Convenience Functions ==============

def generate_club_report(draw_path: str, club_name: str, data: Dict, output: str = None) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate club report from file."""
    draw = DrawStorage.load(draw_path)
    report = ClubReport(draw, data)
    return report.generate_for_club(club_name, output)


def generate_compliance_certificate(draw_path: str, data: Dict, output: str = None) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate compliance certificate from file."""
    draw = DrawStorage.load(draw_path)
    cert = ComplianceCertificate(draw, data)
    return cert.generate(output)


def generate_all_reports(draw_path: str, data: Dict, output_dir: str = "reports") -> None:
    """Generate all reports (clubs, grades, compliance, HTML)."""
    draw = DrawStorage.load(draw_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Club reports
    club_report = ClubReport(draw, data)
    club_report.generate_all_clubs(output_dir)
    
    # Grade reports
    grade_report = GradeReport(draw, data)
    grade_report.generate_all_grades(output_dir)
    
    # Compliance certificate
    cert = ComplianceCertificate(draw, data)
    cert.generate(str(Path(output_dir) / "compliance_certificate.xlsx"))
    
    # HTML report
    generate_html_report(draw, data, str(Path(output_dir) / "draw_report.html"))
    
    print(f"\n✅ All reports generated in {output_dir}/")
