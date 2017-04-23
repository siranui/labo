lazy val root = (project in file("."))
   .settings(
      name := "cifar10"
   )

libraryDependencies ++= Seq(
   "org.scalanlp" %% "breeze" % "0.12",
   "org.scalanlp" %% "breeze-natives" % "0.12",
   "org.scalanlp" %% "breeze-viz" % "0.12"
   )

resolvers ++= Seq(
   "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
   "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
      )
