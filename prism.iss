[Setup]
AppId={{ECCBF40A-00C7-4D62-9E61-F7CBD8BF453A}
AppName=prism
AppVersion=1.3
AppPublisher=geltz
DefaultDirName={autopf}\prism
SetupIconFile=prism.ico
DefaultGroupName=prism
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=.
OutputBaseFilename=prism_setup_1.3
WizardStyle=modern
UninstallDisplayIcon={app}\prism.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\prism\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\prism"; Filename: "{app}\prism.exe"
Name: "{group}\{cm:UninstallProgram,prism}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\prism"; Filename: "{app}\prism.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\prism.exe"; Description: "{cm:LaunchProgram,prism}"; Flags: nowait postinstall skipifsilent