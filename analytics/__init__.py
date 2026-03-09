# analytics/__init__.py
"""
Draw analytics, testing, and reporting tools.

Provides:
- DrawStorage: Pliable JSON format for draw storage with partial import support
- DrawAnalytics: Comprehensive analytics and Excel reports
- DrawTester: Game modification testing with violation checks
- ClubReport: Per-club stakeholder reports
- ComplianceCertificate: Formal constraint compliance verification
- SlotAnalyzer: Unused slot analysis
- DrawVersionManager: Semantic versioning with CHANGELOG
- Rev format export for external system integration
"""

from .storage import (
    DrawStorage, 
    StoredGame, 
    DrawAnalytics, 
    load_draw, 
    analyze_draw,
    export_draw_to_revformat,
    SlotAnalyzer,
    get_slot_analyzer,
)
from .tester import DrawTester, ViolationReport, Violation, test_draw, what_if_move_game
from .reports import (
    ClubReport,
    TeamReport,
    GradeReport,
    ComplianceCertificate,
    generate_html_report,
    generate_club_report,
    generate_compliance_certificate,
    generate_all_reports,
)
from .versioning import DrawVersionManager, DrawVersion, VersionDiff

__all__ = [
    # Storage
    'DrawStorage',
    'StoredGame', 
    'DrawAnalytics',
    'load_draw',
    'analyze_draw',
    'export_draw_to_revformat',
    'SlotAnalyzer',
    'get_slot_analyzer',
    
    # Versioning
    'DrawVersionManager',
    'DrawVersion',
    'VersionDiff',
    
    # Testing
    'DrawTester',
    'ViolationReport',
    'Violation',
    'test_draw',
    'what_if_move_game',
    
    # Reports
    'ClubReport',
    'TeamReport',
    'GradeReport',
    'ComplianceCertificate',
    'generate_html_report',
    'generate_club_report',
    'generate_compliance_certificate',
    'generate_all_reports',
]
